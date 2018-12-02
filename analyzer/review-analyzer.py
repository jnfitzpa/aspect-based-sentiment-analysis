
import sys
import json

review = sys.argv[1]

print(json.dumps({'results': review + " reviewed man!!!"}))
