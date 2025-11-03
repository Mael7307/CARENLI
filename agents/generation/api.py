import tenacity
import yaml
import re
from typing import Optional

from openai import AzureOpenAI
import openai

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from base.logger import setup_logging
from prompt.prompt_model import PromptModel

log = setup_logging()

import re
from typing import Optional

from .abstract import GenerativeModel



class AzureGenerator(GenerativeModel):
    def __init__(self, model_name, prompt_model=None, api_key = None, model_version = "2024-02-01", agent_type="azure", endpoint = "https://lunarchatgpt.openai.azure.com/") -> None:
        super().__init__(model_name, prompt_model)
        self.agent_type = agent_type

        self.api_key = api_key
        self.model_version = model_version
        self.endpoint = endpoint
        
        # Initialize Azure OpenAI client
        log.debug("Instantiate AzureOpenAI client for deployment '%s'", model_name)
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=model_version,
            azure_endpoint=endpoint
        )
            
        if prompt_model is None:
            self.prompt_model = PromptModel()
    
    def reset_client(self):
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.model_version,
            azure_endpoint=self.endpoint
        )

    # handle rate limit
    @tenacity.retry(wait=5)
    def completion_with_backoff(self, messages, model_args=None):
        try:
            # messages should already be in OpenAI format: [{"role": "system", "content": "..."}, ...]
            
            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "messages": messages
            }
            
            # Add model arguments if provided
            if model_args:
                for key, value in model_args.items():
                    api_params[key] = value
            
            response = self.client.chat.completions.create(**api_params)
            
            # Create a wrapper object to match existing response format
            class OpenAIResponse:
                def __init__(self, content):
                    self.content = content
            
            return OpenAIResponse(response.choices[0].message.content)
            
        except Exception as e:
            log.error(f'Error: {e}')
            if messages and len(messages) > 0:
                log.info(messages[0].get('content', str(messages[0])))
            raise e
        
    
    @tenacity.retry(wait=5)
    def __call__(self, messages, **model_args: Optional[dict]):
        # messages should already be in OpenAI format: [{"role": "system", "content": "..."}, ...]
        
        # Prepare API parameters
        api_params = {
            "model": self.model_name,
            "messages": messages
        }
        
        # Add model arguments if provided
        if model_args:
            for key, value in model_args.items():
                api_params[key] = value
        
        response = self.client.chat.completions.create(**api_params)
        result = response.choices[0].message.content.strip()

        # post processing
        if "```" in result:
            result = self.extract_code(result)

        
        if 'response_format' in model_args and model_args['response_format'].get('type') == 'json_object':
            try:
                result = self._safe_json(result)
            except:
                result = result
        
        return result

    def generate(self,
                    model_prompt_dir: str,
                    prompt_name: str,
                    prefix: Optional[str] = None,
                    numbered_list: Optional[bool] = False,
                    remove_number: Optional[bool] = False,
                    test: Optional[bool] = False,
                    model_args: Optional[dict] = {},
                    **replacements) -> str:
            """
            Generates a response from the LLM model.

            Parameters:
            - model_prompt_dir (str): The directory of the model prompt.
            - prompt_name (str): The name of the prompt.
            - prefix (str, optional): A prefix to match before the numbered list.
            - numbered_list (bool, optional): If True, the response will be
            extracted as a numbered list.
            - remove_number (bool, optional): If True, the numbered list will
            be cleaned of numbering.
            """
            system_prompt, user_prompt = self.prompt_model.process_prompt(
                model_name=model_prompt_dir,
                prompt_name=prompt_name,
                **replacements
            )
            response = None
            try:
                # Use completion_with_backoff for consistency
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                response = self.completion_with_backoff(
                    messages=messages,
                    model_args=model_args
                )
            except Exception as e:
                log.error("Error during LLM call: %s", e)
                return
            result = response.content.strip()
            if test:
                log.error(result)
            # post processing
            if "```" in result:
                result = self.extract_code(result)

            if prefix and prefix in result:
                parts = result.split(prefix, 1)
                if len(parts) > 1:
                    result = parts[1].strip()

                    if numbered_list and remove_number:
                        result = re.sub(r'\d+\.\s*', '', result)
                        result = '\n'.join(result.splitlines()).strip()
            elif numbered_list:
                result = self.extract_numbered_list(result, None, remove_number)
            
            if 'response_format' in model_args and model_args['response_format'].get('type') == 'json_object':
                try:
                    result = self._safe_json(result)
                except:
                    result = result
            return result


class GeminiGenerator(GenerativeModel):
    def __init__(self, model_name, prompt_model=None, api_key = None) -> None:
        super().__init__(model_name, prompt_model)
        
        # Configure Gemini API 
        genai.configure(api_key=api_key)
        
        log.debug("Instantiate Gemini.")
        self.client = genai.GenerativeModel("gemini-2.5-pro")

        if prompt_model is None:
            self.prompt_model = PromptModel()
    
    def reset_client(self):
        self.client = genai.GenerativeModel("gemini-2.5-pro")
    
    def _extract_text(self, response) -> str:
        """
        Return first text part from the first candidate, or "" if unavailable.
        """
        try:
            cands = getattr(response, "candidates", None)
            if not cands:
                return ""
            cand = cands[0]
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                return ""
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t.strip():
                    return t.strip()
            return ""
        except Exception:
            return ""

    # handle rate limit
    @tenacity.retry(wait=5)
    def completion_with_backoff(self, messages, model_args=None):
        try:
            # Convert messages to Gemini format
            # messages format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            combined_content = ""
            
            for message in messages:
                if message.get('role') == 'system':
                    combined_content += f"System Instructions: {message['content']}\n\n"
                elif message.get('role') == 'user':
                    combined_content += message['content']
                else:
                    combined_content += message['content']
            
            # Configure generation parameters
            generation_config = {}
            if model_args:
                # Map common parameters to Gemini format
                param_mapping = {
                    'temperature': 'temperature',
                    'max_tokens': 'maxOutputTokens',
                    'top_p': 'topP',
                    'top_k': 'topK'
                }
                for key, value in model_args.items():
                    if key in param_mapping:
                        generation_config[param_mapping[key]] = value
                
                # # Handle JSON output format requirement
                # if 'response_format' in model_args and model_args['response_format'].get('type') == 'json_object':
                #     generation_config['response_mime_type'] = 'application/json'
            
            # Use the existing client instance  
            # Generate response
            if generation_config:
                response = self.client.generate_content(
                    combined_content.strip(),
                    generation_config=generation_config,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    }
                )
            else:
                response = self.client.generate_content(combined_content.strip())
            
            output = self._extract_text(response)
            # Create a wrapper object to match existing response format
            class GeminiResponse:
                def __init__(self, content):
                    self.content = content
            
            return GeminiResponse(output)
            
        except Exception as e:
            log.error(f'Error: {e}')
            if messages and len(messages) > 0:
                log.info(messages[0].get('content', str(messages[0])))
            raise e
        
    
    @tenacity.retry(wait=5)
    def __call__(self, messages, **model_args: Optional[dict]):
        # Convert messages to Gemini format
        # messages format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        combined_content = ""
        
        for message in messages:
            if message.get('role') == 'system':
                combined_content += f"System Instructions: {message['content']}\n\n"
            elif message.get('role') == 'user':
                combined_content += message['content']
            else:
                combined_content += message['content']
        
        # Configure generation parameters
        generation_config = {}
        if model_args:
            # Map common parameters to Gemini format
            param_mapping = {
                    'temperature': 'temperature',
                    'max_tokens': 'maxOutputTokens',
                    'top_p': 'topP',
                    'top_k': 'topK'
                }
            for key, value in model_args.items():
                if key in param_mapping:
                    generation_config[param_mapping[key]] = value
            
            # # Handle JSON output format requirement
            # if 'response_format' in model_args and model_args['response_format'].get('type') == 'json_object':
            #     generation_config['response_mime_type'] = 'application/json'
        
        # Use the existing client instance
        # Generate response
        try:
            if generation_config:
                response = self.client.generate_content(
                    combined_content.strip(),
                    generation_config=generation_config,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    }
                )
            else:
                response = self.client.generate_content(combined_content.strip())
            
            output = self._extract_text(response.text.strip())
        except Exception as e:
            log.error(f'Error: {e}')
            if messages and len(messages) > 0:
                log.info(messages[0].get('content', str(messages[0])))
            raise e
        return output

    def generate(self,
                    model_prompt_dir: str,
                    prompt_name: str,
                    prefix: Optional[str] = None,
                    numbered_list: Optional[bool] = False,
                    remove_number: Optional[bool] = False,
                    test: Optional[bool] = False,
                    model_args: Optional[dict] = {},
                    **replacements) -> str:
            """
            Generates a response from the LLM model.

            Parameters:
            - model_prompt_dir (str): The directory of the model prompt.
            - prompt_name (str): The name of the prompt.
            - prefix (str, optional): A prefix to match before the numbered list.
            - numbered_list (bool, optional): If True, the response will be
            extracted as a numbered list.
            - remove_number (bool, optional): If True, the numbered list will
            be cleaned of numbering.
            """
            system_prompt, user_prompt = self.prompt_model.process_prompt(
                model_name=model_prompt_dir,
                prompt_name=prompt_name,
                **replacements
            )
            response = None
            try:
                # Use completion_with_backoff for consistency
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                response = self.completion_with_backoff(
                    messages=messages,
                    model_args=model_args
                )
            except Exception as e:
                log.error("Error during LLM call: %s", e)
                return
            result = response.content.strip()
            if test:
                log.error(result)
            # post processing
            if "```" in result:
                result = self.extract_code(result)

            if prefix and prefix in result:
                parts = result.split(prefix, 1)
                if len(parts) > 1:
                    result = parts[1].strip()

                    if numbered_list and remove_number:
                        result = re.sub(r'\d+\.\s*', '', result)
                        result = '\n'.join(result.splitlines()).strip()
            elif numbered_list:
                result = self.extract_numbered_list(result, None, remove_number)
            
            if 'response_format' in model_args and model_args['response_format'].get('type') == 'json_object':
                try:
                    result = self._safe_json(result)
                except:
                    result = result
            return result



