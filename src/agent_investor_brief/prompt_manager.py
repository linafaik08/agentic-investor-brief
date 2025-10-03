"""
Universal Prompt Version Manager - Works with any prompt type and version
"""
import mlflow
import os
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

class PromptVersionManager:
    """Universal prompt manager that can handle any prompt type and version from files"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the universal prompt manager
        
        Args:
            prompts_dir: Base directory containing prompt files
        """
        self.prompts_dir = Path(prompts_dir)
        self._cache = {}  # Cache loaded prompts for performance
        
        # Ensure prompts directory exists
        self.prompts_dir.mkdir(exist_ok=True)
        
        logging.info(f"PromptVersionManager initialized with directory: {self.prompts_dir}")
    
    def load_prompt(self, prompt_type: str, version: str = "v1") -> Dict[str, Any]:
        """
        Load any prompt type and version from file
        
        Args:
            prompt_type: Type of prompt (e.g., 'analysis', 'brief', 'summary', 'classification')
            version: Version of the prompt (e.g., 'v1', 'v2', 'experimental')
            
        Returns:
            Dictionary containing all variables from the prompt file
            
        Example:
            prompts = manager.load_prompt("analysis", "v2")
            system_prompt = prompts["SYSTEM_PROMPT"]
            user_template = prompts["USER_TEMPLATE"]
        """
        cache_key = f"{prompt_type}_{version}"
        
        # Return cached version if available
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self.prompts_dir / prompt_type / f"{version}.py"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {file_path}. "
                f"Available: {self.list_versions().get(prompt_type, [])}"
            )
        
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(cache_key, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Extract all uppercase variables from the module
            prompt_data = {
                attr: getattr(module, attr)
                for attr in dir(module)
                if not attr.startswith('_') and attr.isupper()
            }
            
            # Cache the result
            self._cache[cache_key] = prompt_data
            
            logging.info(f"Loaded prompt: {prompt_type}/{version} with keys: {list(prompt_data.keys())}")
            return prompt_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt {prompt_type}/{version}: {str(e)}")
    
    def get_system_prompt(self, prompt_type: str, version: str = "v1") -> str:
        """Get system prompt for specified type and version"""
        prompt_data = self.load_prompt(prompt_type, version)
        return prompt_data.get("SYSTEM_PROMPT", "")
    
    def get_user_template(self, prompt_type: str, version: str = "v1") -> str:
        """Get user template for specified type and version"""
        prompt_data = self.load_prompt(prompt_type, version)
        return prompt_data.get("USER_TEMPLATE", "")
    
    def get_prompt_config(self, prompt_type: str, version: str = "v1") -> Dict[str, str]:
        """Get standard system/user_template config (backward compatibility)"""
        prompt_data = self.load_prompt(prompt_type, version)
        return {
            "system": prompt_data.get("SYSTEM_PROMPT", ""),
            "user_template": prompt_data.get("USER_TEMPLATE", "")
        }
    
    def format_prompt(self, prompt_type: str, version: str = "v1", **kwargs) -> Dict[str, str]:
        """
        Load and format prompts with provided variables
        
        Args:
            prompt_type: Type of prompt
            version: Version of prompt
            **kwargs: Variables to format into the templates
            
        Returns:
            Dictionary with formatted 'system' and 'user' prompts
        """
        prompt_data = self.load_prompt(prompt_type, version)
        
        # Format system prompt if it exists and has format placeholders
        system_prompt = prompt_data.get("SYSTEM_PROMPT", "")
        if system_prompt and kwargs:
            try:
                system_prompt = system_prompt.format(**kwargs)
            except KeyError as e:
                logging.warning(f"Missing variable for system prompt formatting: {e}")
        
        # Format user template
        user_template = prompt_data.get("USER_TEMPLATE", "")
        if user_template and kwargs:
            try:
                user_prompt = user_template.format(**kwargs)
            except KeyError as e:
                logging.warning(f"Missing variable for user template formatting: {e}")
                user_prompt = user_template
        else:
            user_prompt = user_template
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def list_versions(self) -> Dict[str, List[str]]:
        """
        List all available prompt types and their versions
        
        Returns:
            Dictionary mapping prompt types to list of available versions
        """
        versions = {}
        
        if not self.prompts_dir.exists():
            return versions
        
        # Scan all subdirectories for prompt types
        for prompt_type_dir in self.prompts_dir.iterdir():
            if prompt_type_dir.is_dir():
                prompt_type = prompt_type_dir.name
                
                # Find all version files
                version_files = list(prompt_type_dir.glob("*.py"))
                version_names = [f.stem for f in version_files if not f.name.startswith('_')]
                
                if version_names:
                    versions[prompt_type] = sorted(version_names)
        
        return versions
    
    def list_prompt_types(self) -> List[str]:
        """List all available prompt types"""
        return list(self.list_versions().keys())
    
    def create_prompt_template(self, prompt_type: str, version: str = "v1", 
                             system_prompt: str = "", user_template: str = "", 
                             **additional_vars) -> str:
        """
        Create a new prompt file template
        
        Args:
            prompt_type: Type of prompt
            version: Version name
            system_prompt: System prompt content
            user_template: User template content
            **additional_vars: Any additional variables to include
            
        Returns:
            Path to created file
        """
        # Create prompt type directory if it doesn't exist
        prompt_dir = self.prompts_dir / prompt_type
        prompt_dir.mkdir(exist_ok=True)
        
        file_path = prompt_dir / f"{version}.py"
        
        # Generate file content
        content_lines = [
            '"""',
            f'Prompt: {prompt_type}/{version}',
            f'Created: {os.popen("date").read().strip()}',
            '"""',
            '',
            f'SYSTEM_PROMPT = """{system_prompt}"""',
            '',
            f'USER_TEMPLATE = """{user_template}"""'
        ]
        
        # Add any additional variables
        for var_name, var_value in additional_vars.items():
            if var_name.isupper():
                content_lines.extend(['', f'{var_name} = """{var_value}"""'])
        
        # Write file
        with open(file_path, 'w') as f:
            f.write('\n'.join(content_lines))
        
        logging.info(f"Created prompt template: {file_path}")
        return str(file_path)
    
    def validate_prompt(self, prompt_type: str, version: str = "v1", 
                       required_vars: List[str] = None) -> Dict[str, Any]:
        """
        Validate a prompt has required variables and format placeholders
        
        Args:
            prompt_type: Type of prompt
            version: Version to validate
            required_vars: List of required variable names that should be present
            
        Returns:
            Validation results with any issues found
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "missing_vars": [],
            "format_vars_found": [],
            "file_path": str(self.prompts_dir / prompt_type / f"{version}.py")
        }
        
        try:
            prompt_data = self.load_prompt(prompt_type, version)
            
            # Check for required variables
            if required_vars:
                for var in required_vars:
                    if var not in prompt_data:
                        validation_result["missing_vars"].append(var)
                        validation_result["valid"] = False
            
            # Extract format variables from templates
            import re
            system_prompt = prompt_data.get("SYSTEM_PROMPT", "")
            user_template = prompt_data.get("USER_TEMPLATE", "")
            
            # Find format placeholders like {variable_name}
            format_vars = set()
            for text in [system_prompt, user_template]:
                format_vars.update(re.findall(r'\{(\w+)\}', text))
            
            validation_result["format_vars_found"] = list(format_vars)
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Failed to load: {str(e)}")
        
        return validation_result
    
    def clear_cache(self):
        """Clear the prompt cache to force reloading from files"""
        self._cache.clear()
        logging.info("Prompt cache cleared")
    
    def reload_prompt(self, prompt_type: str, version: str = "v1"):
        """Force reload a specific prompt from file"""
        cache_key = f"{prompt_type}_{version}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        return self.load_prompt(prompt_type, version)
    
    def register_prompt_in_mlflow(self, prompt_type: str, version: str = "v1", author: str = None, tags: Dict[str, str] = None) -> str:
        """
        Register prompt in MLflow for tracking and versioning
        
        Args:
            prompt_type: Type of prompt (e.g., 'analysis', 'brief')
            version: Version of the prompt
            author: Author email/name
            tags: Additional tags for the prompt
            
        Returns:
            MLflow prompt name that was registered
        """
        try:
            prompt_data = self.load_prompt(prompt_type, version)
            
            # Create MLflow template with double curly braces
            system_prompt = prompt_data.get("SYSTEM_PROMPT", "")
            user_template = prompt_data.get("USER_TEMPLATE", "")
            
            # Convert single braces to double braces for MLflow
            mlflow_template = f"""SYSTEM: {system_prompt}

USER: {user_template}"""
            
            # Convert {variable} to {{variable}} for MLflow format
            import re
            mlflow_template = re.sub(r'\{([^}]+)\}', r'{{\1}}', mlflow_template)
            
            # Set default tags
            default_tags = {
                "prompt_type": prompt_type,
                "version": version,
                "file_version": version,
                "agent": "investor_agent"
            }
            
            if author:
                default_tags["author"] = author
            
            if tags:
                default_tags.update(tags)
            
            # MLflow prompt name
            mlflow_prompt_name = f"investor-{prompt_type}-prompt"
            
            # Register prompt in MLflow
            prompt = mlflow.genai.register_prompt(
                name=mlflow_prompt_name,
                template=mlflow_template,
                commit_message=f"Registered {prompt_type} prompt version {version}",
                tags=default_tags
            )
            
            logging.info(f"✅ Registered MLflow prompt '{prompt.name}' (version {prompt.version})")
            return mlflow_prompt_name
            
        except Exception as e:
            logging.error(f"Failed to register prompt in MLflow: {str(e)}")
            raise

    def register_all_prompts_in_mlflow(self, author: str = None) -> Dict[str, str]:
        """
        Register all available prompts in MLflow
        
        Args:
            author: Author email/name for all prompts
            
        Returns:
            Dictionary mapping prompt_type_version to MLflow prompt name
        """
        registered_prompts = {}
        versions = self.list_versions()
        
        for prompt_type, version_list in versions.items():
            for version in version_list:
                try:
                    mlflow_name = self.register_prompt_in_mlflow(
                        prompt_type, version, author,
                        tags={"batch_registered": "true"}
                    )
                    registered_prompts[f"{prompt_type}_{version}"] = mlflow_name
                    
                except Exception as e:
                    logging.error(f"Failed to register {prompt_type}/{version}: {str(e)}")
        
        logging.info(f"Registered {len(registered_prompts)} prompts in MLflow")
        return registered_prompts


# Example Usage
if __name__ == "__main__":
    # Initialize manager
    manager = PromptVersionManager("prompts")
    
    # List all available prompts
    print("Available prompts:")
    versions = manager.list_versions()
    for prompt_type, version_list in versions.items():
        print(f"  {prompt_type}: {version_list}")
    
    # Load any prompt
    try:
        analysis_prompt = manager.load_prompt("analysis", "v1")
        print(f"\nLoaded analysis/v1: {list(analysis_prompt.keys())}")
        
        # Format a prompt with variables
        formatted = manager.format_prompt(
            "analysis", "v1",
            ticker="AAPL",
            industry_data="Tech sector data...",
            financial_data="Revenue: $365B..."
        )
        print(f"Formatted prompt ready: {len(formatted['user'])} chars")
        
    except FileNotFoundError as e:
        print(f"Prompt not found: {e}")