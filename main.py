from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

from name_matcher import NameMatcher
from ai_recipe_generator import AdvancedAIRecipeGenerator

app = FastAPI(title="AI Assignment API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI services
name_matcher = NameMatcher()
ai_recipe_generator = AdvancedAIRecipeGenerator()

# Request/Response models
class NameMatchRequest(BaseModel):
    name: str

class NameMatchResponse(BaseModel):
    best_match: Dict[str, float]
    all_matches: List[Dict[str, float]]

class RecipeRequest(BaseModel):
    ingredients: List[str]

class RecipeResponse(BaseModel):
    recipe: str
    ingredients_used: List[str]
    recipe_type: str = "basic"
    similarity_score: float = 0.0

@app.get("/")
async def root():
    return {"message": "AI Assignment API - Name Matching & Recipe Chatbot"}

@app.post("/api/match-name", response_model=NameMatchResponse)
async def match_name(request: NameMatchRequest):
    try:
        result = name_matcher.find_similar_names(request.name)
        return NameMatchResponse(
            best_match=result["best_match"],
            all_matches=result["all_matches"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/get-recipe", response_model=RecipeResponse)
async def get_recipe(request: RecipeRequest):
    try:
        # Use AI-powered recipe generation
        result = ai_recipe_generator.get_recipe_recommendation(request.ingredients)
        return RecipeResponse(
            recipe=result["recipe"],
            ingredients_used=result["ingredients_used"],
            recipe_type=result.get("recipe_type", "ai_generated"),
            similarity_score=result.get("similarity_score", 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI recipe generation failed: {str(e)}")

@app.post("/api/get-creative-recipe", response_model=RecipeResponse)
async def get_creative_recipe(request: RecipeRequest):
    try:
        # Use AI for creative recipe suggestions
        result = ai_recipe_generator.generate_creative_suggestions(request.ingredients)
        return RecipeResponse(
            recipe=result["recipe"],
            ingredients_used=result["ingredients_used"],
            recipe_type=result.get("recipe_type", "creative_ai"),
            similarity_score=0.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Creative AI generation failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    ai_status = "initialized" if ai_recipe_generator.is_initialized else "fallback_mode"
    return {
        "status": "healthy", 
        "services": ["name_matcher", "ai_recipe_generator"],
        "ai_status": ai_status,
        "ai_model": getattr(ai_recipe_generator, 'model_name', 'unknown')
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 