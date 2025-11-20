// workaround gradio after 4.7, not applying any @media rules form the custom .css file

() => {
    console.log(`innerWidth: ${window.innerWidth}` )

    // 1536px rules

    const mediaQuery1536 = window.matchMedia('(min-width: 1536px)')

    function handleWidth1536(event) {

        // display in full width for desktop devices
        document.querySelectorAll(".gradio-container")
            .forEach( (node) => {
                if (event.matches) {
                    node.classList.add("gradio-container-size-full");
                } else {
                    node.classList.remove("gradio-container-size-full")
                }
            });
    }

    mediaQuery1536.addEventListener("change", handleWidth1536);
    mediaQuery1536.dispatchEvent(new MediaQueryListEvent("change", {matches: window.innerWidth >= 1536}));

    // 1921px rules

    const mediaQuery1921 = window.matchMedia('(min-width: 1921px)')

    function handleWidth1921(event) {

        /* Force a 768px_height + 4px_margin_height + navbar_height for the gallery */
        /* Limit height to 768px_height + 2px_margin_height for the thumbnails */
        document.querySelectorAll("#gallery")
            .forEach( (node) => {
                if (event.matches) {
                    node.classList.add("gallery-force-height768");
                    node.classList.add("gallery-limit-height768");
                } else {
                    node.classList.remove("gallery-force-height768");
                    node.classList.remove("gallery-limit-height768");
                }
            });
    }

    mediaQuery1921.addEventListener("change", handleWidth1921);
    mediaQuery1921.dispatchEvent(new MediaQueryListEvent("change", {matches: window.innerWidth >= 1921}));

}
