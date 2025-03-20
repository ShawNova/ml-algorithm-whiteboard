from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json

# Define a Pydantic model
class UserInfo(BaseModel):
    name: str
    age: int

app = FastAPI()

db = {}

@app.get("/get_user/{username}")
def create_get_response(username: str):
    print(username)
    if username in db:
        return json.dumps({"age": f"{db[username]}", "name": username})
    else:
        return json.dumps({"error": "User not found"})

@app.post("/create_user/")
def create_post_response(user: UserInfo):
    print(user)
    db[user.name] = user.age
    return json.dumps({"name": user.name, "age": user.age})

@app.delete("/delete_user/{username}")
def delete_user(username: str):
    if username in db:
        del db[username]
        return json.dumps({"message": f"User {username} deleted"})
    else:
        return json.dumps({"error": "User not found"})

@app.get("/list_users/")
def list_users():
    return json.dumps({"users": list(db.keys())})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # curl -X GET "http://127.0.0.1:8000/get_user/John"
    # response = {"error": "User not found"}
