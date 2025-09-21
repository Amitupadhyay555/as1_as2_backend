import os
try:
    import torch  # Optional at runtime
except Exception:  # pragma: no cover
    torch = None
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import requests
import json
from typing import List, Dict, Optional, Tuple
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import functools
import re
from datetime import datetime
import gc

# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def timeout_decorator(timeout_duration=30):
    """Decorator to add timeout functionality to methods"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_duration)
                except TimeoutError:
                    logger.warning(f"Function {func.__name__} timed out after {timeout_duration}s")
                    raise TimeoutError(f"AI generation timed out")
        return wrapper
    return decorator

class ModelManager:
    """Manages model lifecycle and memory optimization"""
    
    def __init__(self):
        self.models = {}
        self.model_usage = {}
        self.max_memory_usage = 0.8  # 80% of available VRAM
    
    def load_model(self, model_name: str, model_type: str = "seq2seq"):
        """Load model with memory management"""
        if model_name in self.models:
            self.model_usage[model_name] = time.time()
            return self.models[model_name]
        
        try:
            # Clear unused models if memory is tight
            self._cleanup_unused_models()
            
            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            cuda_available = bool(torch and hasattr(torch, "cuda") and torch.cuda.is_available())
            dtype = (torch.float16 if cuda_available else (torch.float32 if torch else None))

            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto" if cuda_available else None,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto" if cuda_available else None,
                    low_cpu_mem_usage=True
                )
            
            self.models[model_name] = {"tokenizer": tokenizer, "model": model}
            self.model_usage[model_name] = time.time()
            
            logger.info(f"✅ Model {model_name} loaded successfully")
            return self.models[model_name]
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None
    
    def _cleanup_unused_models(self):
        """Remove unused models to free memory"""
        if len(self.models) < 2:
            return
            
        # Remove models not used in last 10 minutes
        current_time = time.time()
        to_remove = []
        
        for model_name, last_used in self.model_usage.items():
            if current_time - last_used > 600:  # 10 minutes
                to_remove.append(model_name)
        
        for model_name in to_remove:
            if model_name in self.models:
                del self.models[model_name]
                del self.model_usage[model_name]
                try:
                    if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
                logger.info(f"Cleaned up unused model: {model_name}")

class RecipeTemplate:
    """Enhanced recipe templates and formatting"""
    
    @staticmethod
    def format_recipe(recipe_data: Dict) -> str:
        """Format recipe with proper structure"""
        # Define default strings outside f-string to avoid backslash issues
        default_ingredients = "• Salt and pepper to taste\n• Cooking oil as needed"
        default_tips = "• Taste and adjust seasoning as you cook\n• Do not overcook vegetables to retain nutrients"
        
        template = f"""🍳 **{recipe_data.get('name', 'AI-Generated Recipe')}**

**Cuisine Style:** {recipe_data.get('cuisine', 'International')}
**Cooking Method:** {recipe_data.get('method', 'Pan-cooking')}
**Prep Time:** {recipe_data.get('prep_time', '15 mins')}
**Cook Time:** {recipe_data.get('cook_time', '20 mins')}
**Servings:** {recipe_data.get('servings', '2-3')}
**Difficulty:** {recipe_data.get('difficulty', 'Medium')} ⭐

**Your Ingredients:**
{recipe_data.get('ingredients_display', 'Various ingredients')}

**Additional Ingredients Needed:**
{recipe_data.get('additional_ingredients', default_ingredients)}

**Instructions:**
{recipe_data.get('instructions', 'Follow basic cooking principles')}

**Chef's Tips:**
{recipe_data.get('tips', default_tips)}

**Nutritional Benefits:**
{recipe_data.get('nutrition_info', 'This recipe provides a balanced mix of nutrients from your selected ingredients.')}

---
🤖 **Generated by:** Enhanced AI Recipe System
⏰ **Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        return template.strip()

class AdvancedAIRecipeGenerator:
    def __init__(self):
        """Initialize the enhanced AI-powered recipe generator"""
        self.model_manager = ModelManager()
        self.recipe_template = RecipeTemplate()
        self.is_initialized = False
        self.fallback_mode = False
        self.disable_models = str(os.getenv("AI_DISABLE_MODELS", "false")).lower() in ("1", "true", "yes")
        
        # Enhanced model configurations
        self.model_configs = {
            "primary": {
                "name": "google/flan-t5-base",
                "type": "seq2seq",
                "use_case": "structured_generation"
            },
            "creative": {
                "name": "microsoft/DialoGPT-small",  # Using small for reliability
                "type": "causal",
                "use_case": "creative_suggestions"
            },
            "fallback": {
                "name": "google/flan-t5-small",
                "type": "seq2seq",
                "use_case": "lightweight_generation"
            }
        }
        
        # Enhanced knowledge base
        self.ingredient_database = self._build_ingredient_database()
        self.recipe_patterns = self._build_recipe_patterns()
        self.nutritional_info = self._build_nutritional_database()
        
        logger.info("🚀 Initializing Enhanced AI Recipe Generator...")
        if self.disable_models:
            # Skip loading large models in constrained environments
            logger.warning("AI model initialization disabled via AI_DISABLE_MODELS. Using rule-based fallback.")
            self.is_initialized = False
            self.fallback_mode = True
        else:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models with error handling and fallback"""
        try:
            # Try to load primary model
            primary_config = self.model_configs["primary"]
            self.primary_model = self.model_manager.load_model(
                primary_config["name"], 
                primary_config["type"]
            )
            
            if self.primary_model:
                self.primary_pipeline = pipeline(
                    "text2text-generation",
                    model=self.primary_model["model"],
                    tokenizer=self.primary_model["tokenizer"],
                    max_length=600,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    early_stopping=True
                )
                self.is_initialized = True
                logger.info("✅ Primary model initialized successfully")
            else:
                raise Exception("Failed to load primary model")
                
        except Exception as e:
            logger.warning(f"Primary model initialization failed: {e}")
            self._initialize_fallback_mode()
    
    def _initialize_fallback_mode(self):
        """Initialize fallback mode with lightweight model"""
        try:
            fallback_config = self.model_configs["fallback"]
            self.fallback_model = self.model_manager.load_model(
                fallback_config["name"],
                fallback_config["type"]
            )
            
            if self.fallback_model:
                self.fallback_pipeline = pipeline(
                    "text2text-generation",
                    model=self.fallback_model["model"],
                    tokenizer=self.fallback_model["tokenizer"],
                    max_length=300,
                    temperature=0.6
                )
                self.fallback_mode = True
                self.is_initialized = True
                logger.info("✅ Fallback mode initialized")
            else:
                logger.warning("❌ All AI models failed to load - using rule-based generation")
                self.is_initialized = False
                
        except Exception as e:
            logger.error(f"Fallback initialization failed: {e}")
            self.is_initialized = False
    
    def _build_ingredient_database(self) -> Dict:
        """Build comprehensive ingredient database with properties"""
        return {
            "proteins": {
                "chicken": {"cook_time": "20-25", "method": "pan-fry", "pairs_with": ["garlic", "herbs", "lemon"]},
                "beef": {"cook_time": "15-30", "method": "grill", "pairs_with": ["onions", "peppers", "mushrooms"]},
                "fish": {"cook_time": "10-15", "method": "pan-fry", "pairs_with": ["lemon", "herbs", "olive oil"]},
                "eggs": {"cook_time": "5-10", "method": "scramble", "pairs_with": ["cheese", "vegetables", "herbs"]},
                "tofu": {"cook_time": "10-15", "method": "stir-fry", "pairs_with": ["soy sauce", "ginger", "vegetables"]}
            },
            "vegetables": {
                "onions": {"cook_time": "5-8", "method": "sauté", "essential": True},
                "garlic": {"cook_time": "2-3", "method": "sauté", "essential": True},
                "tomatoes": {"cook_time": "10-15", "method": "simmer", "pairs_with": ["basil", "garlic", "olive oil"]},
                "mushrooms": {"cook_time": "5-8", "method": "sauté", "pairs_with": ["garlic", "herbs", "butter"]},
                "bell peppers": {"cook_time": "8-10", "method": "stir-fry", "pairs_with": ["onions", "garlic"]}
            },
            "grains": {
                "rice": {"cook_time": "18-20", "method": "boil", "ratio": "1:2"},
                "pasta": {"cook_time": "8-12", "method": "boil", "pairs_with": ["tomato", "cheese", "herbs"]},
                "quinoa": {"cook_time": "15", "method": "simmer", "ratio": "1:2"}
            }
        }
    
    def _build_recipe_patterns(self) -> Dict:
        """Build recipe patterns for different cooking styles"""
        return {
            "stir_fry": {
                "steps": ["heat oil", "add aromatics", "add proteins", "add vegetables", "add sauce", "toss and serve"],
                "time_total": "15-20 minutes",
                "difficulty": "Easy"
            },
            "pasta_dish": {
                "steps": ["boil pasta", "prepare sauce", "combine", "add toppings", "serve hot"],
                "time_total": "20-25 minutes", 
                "difficulty": "Easy"
            },
            "rice_bowl": {
                "steps": ["cook rice", "prepare toppings", "season", "assemble bowl", "garnish"],
                "time_total": "25-30 minutes",
                "difficulty": "Medium"
            }
        }
    
    def _build_nutritional_database(self) -> Dict:
        """Build nutritional information database"""
        return {
            "chicken": "High in protein, B vitamins, and selenium",
            "vegetables": "Rich in vitamins, minerals, and fiber", 
            "rice": "Good source of carbohydrates and energy",
            "eggs": "Complete protein with essential amino acids",
            "olive oil": "Healthy monounsaturated fats and vitamin E"
        }
    
    @timeout_decorator(25)
    def generate_recipe_with_ai(self, ingredients: List[str]) -> Dict:
        """Generate comprehensive recipe using AI with timeout protection"""
        if not ingredients:
            return self._create_error_response("No ingredients provided")
        
        if not self.is_initialized:
            return self._generate_intelligent_fallback(ingredients)
        
        try:
            # Analyze ingredients first
            ingredient_analysis = self._analyze_ingredients(ingredients)
            
            # Create enhanced prompt
            prompt = self._create_enhanced_prompt(ingredients, ingredient_analysis)
            
            # Generate with appropriate model
            if hasattr(self, 'primary_pipeline') and not self.fallback_mode:
                generated_content = self._generate_with_primary_model(prompt)
            else:
                generated_content = self._generate_with_fallback_model(prompt)
            
            # Process and structure the generated content
            recipe_data = self._process_generated_content(
                generated_content, ingredients, ingredient_analysis
            )
            
            # Format final recipe
            formatted_recipe = self.recipe_template.format_recipe(recipe_data)
            
            return {
                "recipe": formatted_recipe,
                "ingredients_used": ingredients,
                "recipe_type": "ai_enhanced",
                "generation_method": "primary" if not self.fallback_mode else "fallback",
                "analysis": ingredient_analysis,
                "generation_time": time.time(),
                "success": True
            }
            
        except TimeoutError:
            logger.warning("AI generation timed out, using intelligent fallback")
            return self._generate_intelligent_fallback(ingredients)
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return self._generate_intelligent_fallback(ingredients)
    
    def _analyze_ingredients(self, ingredients: List[str]) -> Dict:
        """Analyze ingredients to determine cooking approach"""
        analysis = {
            "proteins": [],
            "vegetables": [],
            "grains": [],
            "seasonings": [],
            "cooking_method": "pan-cooking",
            "cuisine_style": "international",
            "estimated_time": "20-25 minutes",
            "difficulty": "Medium"
        }
        
        ingredients_lower = [ing.lower().strip() for ing in ingredients]
        
        # Categorize ingredients
        for ingredient in ingredients_lower:
            for category, items in self.ingredient_database.items():
                for item_name in items.keys():
                    if item_name in ingredient or ingredient in item_name:
                        analysis[category].append(ingredient)
                        break
        
        # Determine cooking method based on ingredients
        if analysis["grains"]:
            if any("pasta" in grain for grain in analysis["grains"]):
                analysis["cooking_method"] = "boiling and sautéing"
                analysis["cuisine_style"] = "italian"
            elif any("rice" in grain for grain in analysis["grains"]):
                analysis["cooking_method"] = "steaming and stir-frying"
                analysis["cuisine_style"] = "asian"
        elif analysis["proteins"]:
            analysis["cooking_method"] = "pan-frying"
        
        # Estimate cooking time
        max_time = 15
        for protein in analysis["proteins"]:
            if protein in self.ingredient_database.get("proteins", {}):
                time_range = self.ingredient_database["proteins"][protein].get("cook_time", "15-20")
                max_time = max(max_time, int(time_range.split("-")[-1]))
        
        analysis["estimated_time"] = f"{max_time + 5}-{max_time + 10} minutes"
        
        return analysis
    
    def _create_enhanced_prompt(self, ingredients: List[str], analysis: Dict) -> str:
        """Create sophisticated prompt for AI generation"""
        ingredients_str = ", ".join(ingredients)
        
        prompt = f"""Create a detailed recipe using these ingredients: {ingredients_str}

Recipe requirements:
- Cuisine style: {analysis['cuisine_style']}
- Cooking method: {analysis['cooking_method']}
- Estimated time: {analysis['estimated_time']}
- Include: recipe name, ingredient quantities, step-by-step instructions, cooking tips

Generate a complete recipe with:
1. Creative recipe name
2. Ingredient list with measurements
3. Clear cooking instructions (numbered steps)
4. Cooking time and temperature details
5. Serving suggestions

Recipe:"""
        return prompt
    
    def _generate_with_primary_model(self, prompt: str) -> str:
        """Generate content using primary AI model"""
        try:
            result = self.primary_pipeline(
                prompt,
                max_length=500,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            return result[0]['generated_text']
        except Exception as e:
            logger.error(f"Primary model generation failed: {e}")
            raise
    
    def _generate_with_fallback_model(self, prompt: str) -> str:
        """Generate content using fallback model"""
        try:
            result = self.fallback_pipeline(
                prompt,
                max_length=300,
                num_return_sequences=1,
                temperature=0.6
            )
            return result[0]['generated_text']
        except Exception as e:
            logger.error(f"Fallback model generation failed: {e}")
            raise
    
    def _process_generated_content(self, content: str, ingredients: List[str], analysis: Dict) -> Dict:
        """Process and structure AI-generated content"""
        # Extract recipe components using regex and heuristics
        lines = content.split('\n')
        
        recipe_data = {
            "name": self._extract_recipe_name(content, ingredients),
            "cuisine": analysis['cuisine_style'].title(),
            "method": analysis['cooking_method'].title(),
            "prep_time": "10-15 mins",
            "cook_time": analysis['estimated_time'],
            "servings": "2-3 people",
            "difficulty": analysis['difficulty'],
            "ingredients_display": self._format_ingredients_display(ingredients),
            "additional_ingredients": self._suggest_additional_ingredients(ingredients, analysis),
            "instructions": self._extract_or_generate_instructions(content, analysis),
            "tips": self._generate_cooking_tips(ingredients, analysis),
            "nutrition_info": self._generate_nutrition_info(ingredients)
        }
        
        return recipe_data
    
    def _extract_recipe_name(self, content: str, ingredients: List[str]) -> str:
        """Extract or generate creative recipe name"""
        # Try to extract from AI content first
        lines = content.split('\n')
        for line in lines[:3]:
            if len(line.split()) <= 6 and any(ing.lower() in line.lower() for ing in ingredients):
                return line.strip().title()
        
        # Generate name based on main ingredients
        main_ingredients = ingredients[:2]
        return f"Delicious {' and '.join(main_ingredients).title()} Fusion"
    
    def _format_ingredients_display(self, ingredients: List[str]) -> str:
        """Format ingredients for display"""
        return '\n'.join([f"• {ing.title()}" for ing in ingredients])
    
    def _suggest_additional_ingredients(self, ingredients: List[str], analysis: Dict) -> str:
        """Suggest additional ingredients based on analysis"""
        suggestions = ["• Salt and pepper to taste", "• Cooking oil (2-3 tbsp)"]
        
        # Add cuisine-specific suggestions
        if analysis['cuisine_style'] == 'italian':
            suggestions.append("• Fresh herbs (basil/oregano)")
            suggestions.append("• Parmesan cheese (optional)")
        elif analysis['cuisine_style'] == 'asian':
            suggestions.append("• Soy sauce (2-3 tbsp)")
            suggestions.append("• Fresh ginger (1 inch piece)")
        
        if not any("onion" in ing.lower() for ing in ingredients):
            suggestions.append("• 1 medium onion")
        
        return '\n'.join(suggestions)
    
    def _extract_or_generate_instructions(self, content: str, analysis: Dict) -> str:
        """Extract instructions from AI content or generate structured ones"""
        # Try to extract numbered steps from content
        lines = content.split('\n')
        instruction_lines = []
        
        for line in lines:
            line = line.strip()
            if (line and (line[0].isdigit() or 
                         line.lower().startswith(('step', 'first', 'then', 'next', 'finally')))):
                instruction_lines.append(line)
        
        if instruction_lines:
            return '\n'.join([f"{i+1}. {line.lstrip('0123456789. ')}" 
                            for i, line in enumerate(instruction_lines[:8])])
        
        # Generate structured instructions based on cooking method
        return self._generate_structured_instructions(analysis)
    
    def _generate_structured_instructions(self, analysis: Dict) -> str:
        """Generate structured cooking instructions"""
        method = analysis['cooking_method'].lower()
        
        if "pasta" in method or "boiling" in method:
            return """1. Bring a large pot of salted water to boil
2. Add pasta and cook according to package directions
3. Meanwhile, heat oil in a large pan over medium heat
4. Add aromatics (onions, garlic) and cook until fragrant
5. Add other ingredients and cook until tender
6. Drain pasta and add to the pan
7. Toss everything together and season to taste
8. Serve hot with desired garnishes"""
        
        elif "stir" in method or "fry" in method:
            return """1. Heat oil in a large wok or pan over high heat
2. Add aromatics and stir-fry for 30 seconds
3. Add proteins and cook until nearly done
4. Add harder vegetables first, softer ones later
5. Stir-fry everything for 2-3 minutes
6. Add sauces and seasonings
7. Toss everything together until well coated
8. Serve immediately while hot"""
        
        else:
            return """1. Prepare all ingredients by washing and chopping
2. Heat oil in a large pan over medium heat
3. Start with aromatics (onions, garlic)
4. Add proteins and cook until done
5. Add vegetables in order of cooking time needed
6. Season progressively and taste as you go
7. Cook until all ingredients are tender
8. Adjust seasoning and serve hot"""
    
    def _generate_cooking_tips(self, ingredients: List[str], analysis: Dict) -> str:
        """Generate relevant cooking tips"""
        tips = ["• Don't overcrowd the pan - cook in batches if needed"]
        
        if any("garlic" in ing.lower() for ing in ingredients):
            tips.append("• Add garlic last to prevent burning")
        
        if analysis['proteins']:
            tips.append("• Let proteins rest before cutting for better texture")
        
        if any("vegetable" in ing.lower() for ing in ingredients):
            tips.append("• Cut vegetables uniformly for even cooking")
        
        tips.append("• Taste and adjust seasoning throughout cooking")
        
        return '\n'.join(tips)
    
    def _generate_nutrition_info(self, ingredients: List[str]) -> str:
        """Generate nutritional information"""
        info_parts = []
        
        for ingredient in ingredients[:3]:  # Focus on main ingredients
            ingredient_lower = ingredient.lower()
            for key, info in self.nutritional_info.items():
                if key in ingredient_lower:
                    info_parts.append(f"{ingredient.title()}: {info}")
                    break
        
        if not info_parts:
            info_parts.append("This recipe provides a balanced combination of nutrients from your selected ingredients.")
        
        return ' | '.join(info_parts)
    
    def _generate_intelligent_fallback(self, ingredients: List[str]) -> Dict:
        """Generate high-quality fallback recipe when AI fails"""
        analysis = self._analyze_ingredients(ingredients)
        
        recipe_data = {
            "name": f"Homestyle {' & '.join(ingredients[:2]).title()} Dish",
            "cuisine": analysis['cuisine_style'].title(),
            "method": analysis['cooking_method'].title(),
            "prep_time": "10-15 mins",
            "cook_time": analysis['estimated_time'],
            "servings": "2-3 people",
            "difficulty": analysis['difficulty'],
            "ingredients_display": self._format_ingredients_display(ingredients),
            "additional_ingredients": self._suggest_additional_ingredients(ingredients, analysis),
            "instructions": self._generate_structured_instructions(analysis),
            "tips": self._generate_cooking_tips(ingredients, analysis),
            "nutrition_info": self._generate_nutrition_info(ingredients)
        }
        
        formatted_recipe = self.recipe_template.format_recipe(recipe_data)
        
        return {
            "recipe": formatted_recipe,
            "ingredients_used": ingredients,
            "recipe_type": "intelligent_fallback",
            "generation_method": "rule_based_ai",
            "analysis": analysis,
            "generation_time": time.time(),
            "success": True
        }
    
    def _create_error_response(self, error_msg: str) -> Dict:
        """Create standardized error response"""
        return {
            "recipe": f"❌ Error: {error_msg}\n\nPlease provide valid ingredients to generate a recipe.",
            "ingredients_used": [],
            "recipe_type": "error",
            "success": False,
            "error": error_msg
        }
    
    def get_recipe_recommendation(self, ingredients: List[str]) -> Dict:
        """Main method to get enhanced AI-powered recipe recommendations"""
        if not ingredients:
            return self._create_error_response("No ingredients provided")
        
        # Clean and validate ingredients
        cleaned_ingredients = [ing.strip() for ing in ingredients if ing.strip()]
        if not cleaned_ingredients:
            return self._create_error_response("No valid ingredients provided")
        
        logger.info(f"🍳 Generating recipe for: {', '.join(cleaned_ingredients)}")
        
        try:
            # Generate main recipe
            result = self.generate_recipe_with_ai(cleaned_ingredients)
            
            # Add generation metadata
            result["total_ingredients"] = len(cleaned_ingredients)
            result["generation_timestamp"] = datetime.now().isoformat()
            result["model_status"] = "active" if self.is_initialized else "fallback"
            
            logger.info(f"✅ Recipe generated successfully using {result.get('generation_method', 'unknown')} method")
            return result
            
        except Exception as e:
            logger.error(f"Recipe generation failed: {e}")
            return self._create_error_response(f"Recipe generation failed: {str(e)}")
    
    def get_multiple_recipe_suggestions(self, ingredients: List[str], count: int = 3) -> List[Dict]:
        """Generate multiple recipe variations"""
        if not ingredients or count < 1:
            return []
        
        suggestions = []
        base_analysis = self._analyze_ingredients(ingredients)
        
        # Generate variations by adjusting cooking methods and styles
        variations = [
            {"style": "quick_and_easy", "method": "stir-fry", "time": "15 mins"},
            {"style": "comfort_food", "method": "slow_cook", "time": "30 mins"},
            {"style": "healthy_option", "method": "steam", "time": "20 mins"}
        ]
        
        for i, variation in enumerate(variations[:count]):
            try:
                modified_analysis = base_analysis.copy()
                modified_analysis.update(variation)
                
                suggestion = self._generate_intelligent_fallback(ingredients)
                suggestion["recipe_type"] = f"variation_{i+1}"
                suggestion["variation_theme"] = variation["style"]
                suggestions.append(suggestion)
                
            except Exception as e:
                logger.error(f"Failed to generate variation {i+1}: {e}")
                continue
        
        return suggestions
    
    def health_check(self) -> Dict:
        """Check system health and model status"""
        return {
            "system_status": "healthy" if self.is_initialized else "degraded",
            "models_loaded": len(self.model_manager.models),
            "fallback_mode": self.fallback_mode,
            "memory_usage": (torch.cuda.memory_allocated() if (torch and hasattr(torch, "cuda") and torch.cuda.is_available()) else 0),
            "last_generation": getattr(self, '_last_generation_time', None)
        }

# Usage example and testing
if __name__ == "__main__":
    # Initialize the enhanced generator
    generator = AdvancedAIRecipeGenerator()
    
    # Test with sample ingredients
    test_ingredients = ["chicken", "rice", "vegetables", "garlic"]
    
    print("🧪 Testing Enhanced AI Recipe Generator...")
    result = generator.get_recipe_recommendation(test_ingredients)
    
    if result["success"]:
        print("\n" + "="*50)
        print(result["recipe"])
        print("="*50)
        print(f"\n✅ Generation method: {result['generation_method']}")
        print(f"📊 System status: {generator.health_check()}")
    else:
        print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")