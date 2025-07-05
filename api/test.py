def handler(request, context):
    """Simple test handler for Vercel"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
        },
        'body': '{"message": "Python function is working!", "path": "' + str(request.get('path', 'unknown')) + '"}'
    }