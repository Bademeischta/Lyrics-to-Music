import argparse
from src.api.server import generate_music


def main():
    parser = argparse.ArgumentParser(description='Generate music from lyrics')
    parser.add_argument('lyrics', help='Input lyrics')
    parser.add_argument('--genre_id', type=int, default=0)
    args = parser.parse_args()

    style = {'genre_id': args.genre_id}
    url = generate_music(args.lyrics, style)
    print('Generated file:', url)

if __name__ == '__main__':
    main()
