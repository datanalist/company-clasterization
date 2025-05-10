from fastapi import FastAPI
import uvicorn


app = FastAPI()


@app.get("/{user_id}")
def foo(user_id: str):
    return f"Hello from GET, {user_id} !"


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
