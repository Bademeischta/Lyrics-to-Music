from src.api.server import generate_music


def generate(event, context):
    lyrics = event.get('lyrics', '')
    style = event.get('style', {})
    path = generate_music(lyrics, style)
    return {
        'statusCode': 200,
        'body': {
            'download_url': path,
            'metadata': {'lyrics': lyrics, 'style': style}
        }
    }
