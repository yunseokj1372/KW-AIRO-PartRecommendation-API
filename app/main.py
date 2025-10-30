from fastapi import FastAPI, Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
from app.routers import recommendation
from app.core.config import settings
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="KW AIRO Triage Part Recommendation API",
    description="""
    API for recommending parts based on symptom codes
    """,
    version="0.0.0",
    contact={
        "name": "AIRO Support",
        "email": "support@airo.com"
    }
)

# Define the API key header
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=True,
    description="API key for authentication. Must be provided in the `X-API-Key` header."
)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify the provided API key against the configured secret key

    Args:
        api_key (str): The API key from the request header.

    Returns:
        str: The verified API key.

    Raises:
        HTTPException: If the API key is invalid.
    """
    if api_key != settings.SECRET_KEY:
        raise HTTPException(
            status_code=410,
            detail="Invalid API key"
        )
    return api_key

app.include_router(
    recommendation.router,
    dependencies=[Depends(verify_api_key)]
)

# Root endpoint
@app.get("/",
    dependencies=[Depends(verify_api_key)],
    summary="Root endpoint",
    description="Returns a welcome message to confirm the API is running.",
    response_description="Welcome message",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {"message": "Welcome to AIRO's Part Recommendation app"}
                }
            }
        },
        410: {
            "description": "Invalid API key",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid API key"}
                }
            }
        }
    }
)

def read_root():
    return {"message": "Welcome to AIRO's Part Recommendation app"}

@app.get("/health",
    summary="Health check endpoint",
    description="Returns a health check response to confirm the API is running (No Authentication Required).",
    response_description="Health check response",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {"status": "healthy"}
                }
            }
        }
    }
)

async def health_check():
    """
    Health check endpoint to verify the recommendation API is running
    """
    return {"status": "healthy"}

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication"
        }
    }
    
    # Apply security globally
    openapi_schema["security"] = [{"ApiKeyHeader": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi