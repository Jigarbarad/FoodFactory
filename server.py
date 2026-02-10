from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone, timedelta
import os
import uuid
import httpx
from dotenv import load_dotenv 

load_dotenv()

app = FastAPI(title="AJ's Food Bar Van API")

@app.get("/")
async def root():
    return RedirectResponse(url='/docs')

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
MONGO_URL = os.environ.get("MONGO_URL", "mongodb+srv://jigarbarad1586_db_user:MongoDB1586@cluster0.ro1acqf.mongodb.net/?appName=Cluster0")
DB_NAME = os.environ.get("DB_NAME", "ajs_foodbar")

from types import SimpleNamespace

USING_FAKE_DB = False
try:
    try:
        from pymongo import MongoClient as SyncMongoClient
        sync_client = SyncMongoClient(MONGO_URL, serverSelectionTimeoutMS=2000)
        sync_client.admin.command("ping")
        client = AsyncIOMotorClient(MONGO_URL)
        db = client[DB_NAME]
    except Exception:
        raise
except Exception as e:
    print(f"MongoDB not reachable, using in-memory fallback DB: {e}")
    USING_FAKE_DB = True

    class InMemoryCollection:
        def __init__(self, initial=None):
            self._data = [dict(d) for d in (initial or [])]
        def find(self, *args, **kwargs):
            return self
        async def to_list(self, length):
            return [d.copy() for d in self._data[:length]]
        async def find_one(self, filter, projection=None):
            for d in self._data:
                ok = True
                for k, v in filter.items():
                    if d.get(k) != v:
                        ok = False
                        break
                if ok:
                    return d.copy()
            return None
        async def insert_one(self, doc):
            self._data.append(dict(doc))
            return SimpleNamespace(inserted_id=None)
        async def insert_many(self, docs):
            self._data.extend([dict(d) for d in docs])
            return SimpleNamespace(acknowledged=True)
        async def update_one(self, filter, update):
            for d in self._data:
                ok = True
                for k, v in filter.items():
                    if d.get(k) != v:
                        ok = False
                        break
                if ok:
                    if "$set" in update:
                        d.update(update["$set"])
                    return SimpleNamespace(matched_count=1, modified_count=1)
            return SimpleNamespace(matched_count=0, modified_count=0)
        async def delete_one(self, filter):
            for i, d in enumerate(self._data):
                ok = True
                for k, v in filter.items():
                    if d.get(k) != v:
                        ok = False
                        break
                if ok:
                    self._data.pop(i)
                    return SimpleNamespace(deleted_count=1)
            return SimpleNamespace(deleted_count=0)
        async def count_documents(self, filter):
            if not filter:
                return len(self._data)
            cnt = 0
            for d in self._data:
                ok = True
                for k, v in filter.items():
                    if d.get(k) != v:
                        ok = False
                        break
                if ok:
                    cnt += 1
            return cnt

    seed_menu_items = [
        {"item_id": "item_hotdog01", "name": "Hot Dog", "description": "Classic grilled hot dog", "price": 3.50, "category": "mains", "image_url": "https://images.unsplash.com/photo-1612392062631-94e3f327c7fb?w=400", "available": True},
        {"item_id": "item_burger01", "name": "Hamburger", "description": "Juicy beef patty", "price": 5.00, "category": "mains", "image_url": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400", "available": True},
        {"item_id": "item_donner01", "name": "Donner Meat and Chips", "description": "Donner meat with crispy chips", "price": 7.50, "category": "mains", "image_url": "https://images.unsplash.com/photo-1561651823-34feb02250e4?w=400", "available": True},
        {"item_id": "item_chili01", "name": "Chili Cheese", "description": "Loaded fries with chili", "price": 4.50, "category": "sides", "image_url": "https://images.unsplash.com/photo-1585109649139-366815a0d713?w=400", "available": True},
        {"item_id": "item_wings01", "name": "Spicy Wings", "description": "Crispy chicken wings", "price": 5.50, "category": "sides", "image_url": "https://images.unsplash.com/photo-1608039755401-742074f0548d?w=400", "available": True},
        {"item_id": "item_fries01", "name": "French Fries", "description": "Golden crispy fries", "price": 2.50, "category": "sides", "image_url": "https://images.unsplash.com/photo-1573080496219-bb080dd4f877?w=400", "available": True},
        {"item_id": "item_coke01", "name": "Soft Drink", "description": "Coca-Cola, Sprite, or Fanta", "price": 1.50, "category": "drinks", "image_url": "https://images.unsplash.com/photo-1581636625402-29b2a704ef13?w=400", "available": True}
    ]

    db = SimpleNamespace(
        menu_items=InMemoryCollection(seed_menu_items),
        users=InMemoryCollection([]),
        user_sessions=InMemoryCollection([]),
        orders=InMemoryCollection([]),
    )

class MenuItem(BaseModel):
    item_id: Optional[str] = None
    name: str
    description: str
    price: float
    category: str
    image_url: Optional[str] = None
    available: bool = True

class OrderItem(BaseModel):
    item_id: str
    name: str
    quantity: int
    price: float

class Order(BaseModel):
    order_id: Optional[str] = None
    user_id: Optional[str] = None
    customer_name: str
    customer_phone: str
    customer_address: Optional[str] = None
    order_type: str 
    items: List[OrderItem]
    total: float
    status: str = "pending"
    created_at: Optional[datetime] = None

class User(BaseModel):
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    role: str = "customer"

async def get_current_user(request: Request) -> Optional[User]:
    session_token = request.cookies.get("session_token")
    if not session_token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header.split(" ")[1]
    if not session_token:
        return None
    session = await db.user_sessions.find_one({"session_token": session_token}, {"_id": 0})
    if not session:
        return None
    expires_at = session.get("expires_at")
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        return None
    user = await db.users.find_one({"user_id": session["user_id"]}, {"_id": 0})
    if not user:
        return None
    return User(**user)

async def require_auth(request: Request) -> User:
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

async def require_admin(request: Request) -> User:
    user = await require_auth(request)
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

@app.post("/api/auth/session")
async def exchange_session(request: Request, response: Response):
    body = await request.json()
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": session_id}
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid session")
        data = resp.json()
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    session_token = data.get("session_token")
    existing_user = await db.users.find_one({"email": data["email"]}, {"_id": 0})
    if existing_user:
        user_id = existing_user["user_id"]
        await db.users.update_one({"email": data["email"]}, {"$set": {"name": data["name"], "picture": data.get("picture")}})
    else:
        await db.users.insert_one({"user_id": user_id, "email": data["email"], "name": data["name"], "picture": data.get("picture"), "role": "customer", "created_at": datetime.now(timezone.utc)})
    await db.user_sessions.insert_one({"user_id": user_id, "session_token": session_token, "expires_at": datetime.now(timezone.utc) + timedelta(days=7), "created_at": datetime.now(timezone.utc)})
    response.set_cookie(key="session_token", value=session_token, httponly=True, secure=True, samesite="none", path="/", max_age=7*24*60*60)
    user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    return user

@app.get("/api/auth/me")
async def get_me(user: User = Depends(require_auth)):
    return user

@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    session_token = request.cookies.get("session_token")
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    response.delete_cookie(key="session_token", path="/")
    return {"message": "Logged out"}

@app.post("/api/admin/set-role")
async def set_user_role(request: Request, admin: User = Depends(require_admin)):
    body = await request.json()
    email = body.get("email")
    role = body.get("role", "admin")
    result = await db.users.update_one({"email": email}, {"$set": {"role": role}})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": f"User {email} role set to {role}"}

@app.get("/api/menu")
async def get_menu():
    items = await db.menu_items.find({}, {"_id": 0}).to_list(100)
    return items

@app.get("/api/menu/{item_id}")
async def get_menu_item(item_id: str):
    item = await db.menu_items.find_one({"item_id": item_id}, {"_id": 0})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.post("/api/menu")
async def create_menu_item(item: MenuItem, admin: User = Depends(require_admin)):
    item.item_id = f"item_{uuid.uuid4().hex[:8]}"
    await db.menu_items.insert_one(item.model_dump())
    return item

@app.put("/api/menu/{item_id}")
async def update_menu_item(item_id: str, item: MenuItem, admin: User = Depends(require_admin)):
    item.item_id = item_id
    result = await db.menu_items.update_one({"item_id": item_id}, {"$set": item.model_dump()})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.delete("/api/menu/{item_id}")
async def delete_menu_item(item_id: str, admin: User = Depends(require_admin)):
    result = await db.menu_items.delete_one({"item_id": item_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Item deleted"}

@app.patch("/api/menu/{item_id}/availability")
async def toggle_availability(item_id: str, request: Request, admin: User = Depends(require_admin)):
    body = await request.json()
    available = body.get("available", True)
    result = await db.menu_items.update_one({"item_id": item_id}, {"$set": {"available": available}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Availability updated"}

@app.post("/api/orders")
async def create_order(order: Order, request: Request):
    user = await get_current_user(request)
    order.order_id = f"order_{uuid.uuid4().hex[:8]}"
    order.created_at = datetime.now(timezone.utc)
    if user:
        order.user_id = user.user_id
    await db.orders.insert_one(order.model_dump())
    return order

@app.get("/api/orders")
async def get_orders(admin: User = Depends(require_admin)):
    orders = await db.orders.find({}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return orders

@app.get("/api/orders/my")
async def get_my_orders(user: User = Depends(require_auth)):
    orders = await db.orders.find({"user_id": user.user_id}, {"_id": 0}).sort("created_at", -1).to_list(50)
    return orders

@app.patch("/api/orders/{order_id}/status")
async def update_order_status(order_id: str, request: Request, admin: User = Depends(require_admin)):
    body = await request.json()
    status = body.get("status")
    result = await db.orders.update_one({"order_id": order_id}, {"$set": {"status": status}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"message": "Status updated"}

@app.get("/api/status")
async def get_status():
    now = datetime.now(timezone.utc)
    uk_time = now
    hour = uk_time.hour
    minute = uk_time.minute
    is_open = (hour > 17 or (hour == 17 and minute >= 30)) and hour < 24
    return {"is_open": is_open, "opens_at": "5:30 PM", "closes_at": "12:00 AM", "current_time": uk_time.strftime("%H:%M")}

@app.on_event("startup")
async def seed_menu():
    try:
        count = await db.menu_items.count_documents({})
    except Exception as e:
        print(f"Skipping menu seed, DB not available: {e}")
        return
    if count == 0:
        menu_items = [
            {"item_id": "item_hotdog01", "name": "Hot Dog", "description": "Classic grilled hot dog", "price": 3.50, "category": "mains", "image_url": "https://images.unsplash.com/photo-1612392062631-94e3f327c7fb?w=400", "available": True},
            {"item_id": "item_burger01", "name": "Hamburger", "description": "Juicy beef patty", "price": 5.00, "category": "mains", "image_url": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400", "available": True},
            {"item_id": "item_donner01", "name": "Donner Meat and Chips", "description": "Donner meat with chips", "price": 7.50, "category": "mains", "image_url": "https://images.unsplash.com/photo-1561651823-34feb02250e4?w=400", "available": True},
            {"item_id": "item_chili01", "name": "Chili Cheese", "description": "Loaded fries with chili", "price": 4.50, "category": "sides", "image_url": "https://images.unsplash.com/photo-1585109649139-366815a0d713?w=400", "available": True},
            {"item_id": "item_wings01", "name": "Spicy Wings", "description": "Crispy chicken wings", "price": 5.50, "category": "sides", "image_url": "https://images.unsplash.com/photo-1608039755401-742074f0548d?w=400", "available": True},
            {"item_id": "item_fries01", "name": "French Fries", "description": "Golden crispy fries", "price": 2.50, "category": "sides", "image_url": "https://images.unsplash.com/photo-1573080496219-bb080dd4f877?w=400", "available": True},
            {"item_id": "item_coke01", "name": "Soft Drink", "description": "Soft drink", "price": 1.50, "category": "drinks", "image_url": "https://images.unsplash.com/photo-1581636625402-29b2a704ef13?w=400", "available": True}
        ]
        await db.menu_items.insert_many(menu_items)
        print("Menu seeded!")

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "AJ's Food Bar Van API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)