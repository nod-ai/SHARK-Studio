from apps.stable_diffusion.src import args
from apps.stable_diffusion.scripts import (
    img2img,
    txt2img,
    inpaint,
    outpaint,
)

if __name__ == "__main__":
    if args.app == "txt2img":
        txt2img.main()
    elif args.app == "img2img":
        img2img.main()
    elif args.app == "inpaint":
        inpaint.main()
    elif args.app == "outpaint":
        outpaint.main()
    else:
        print(f"args.app value is {args.app} but this isn't supported")
