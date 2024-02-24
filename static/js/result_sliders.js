// Custom Range Slider edited from https://www.youtube.com/watch?v=gjPllrhIYsM
$(document).ready(function() {
    var sliderContainer = $(".slider-container");

    sliderContainer.each(function() {
        let anxietySlider = $(this).find('.anxiety-slider');
        let fill = $(this).find('.bar .fill');

        function setBar() {
            // convert number out of 21 into percentage
            let scaledFill = (anxietySlider.val() / 21) * 100
            fill.css("width", scaledFill + "%");
        }

        anxietySlider.on("input", setBar);
        console.log(anxietySlider.val());
        setBar();
    })
});