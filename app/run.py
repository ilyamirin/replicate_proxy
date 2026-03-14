import uvicorn

from app.config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port)


if __name__ == "__main__":
    main()
