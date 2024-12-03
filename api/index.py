from sanic import Sanic
from sanic.response import json
app = Sanic()
 
 
@app.route('/api/test')
async def index(request, path=""):
    return json({'hello': '123'})