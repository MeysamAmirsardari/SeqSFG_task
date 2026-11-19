from .cli import main

if __name__ == "__main__":          # guarded: `check` with several seeds starts worker processes,
    main()                          # which import this module again
