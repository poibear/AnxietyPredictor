// WEB CIFAR @ https://www.youtube.com/watch?v=cKTgIDkRsGc for form & js
// using jquery instead of traditional js

$(document).ready(function () {
    // MOVING THROUGH FORM
    const steps = $('.step');
    const nextBtn = $('.next-btn');
    const prevBtn = $('.back-btn');
    const progressBars = $('.progress-bar');

    
    nextBtn.each(function() {
        $(this).on("click", function() {
            changeStep("next");
        })
    })


    prevBtn.each(function() {
        $(this).on("click", function() {
            changeStep("back");
        })
    })


    function changeStep(action) {
        let stepIndex = 0; // start @ 0th step
        let barIndex = 0;

        const inactiveBarClr = $("html").css('--inactive-bar');
        const activeBarClr = $("html").css('--active-bar');

        const stepActive = $("form .step.active");
        const barActive = $(".survey-topic .progress-bar[value='1']");

        stepIndex = steps.index(stepActive);
        barIndex = progressBars.index(barActive);

        // set current step/bar to inactive before changing next current one
        $(steps[stepIndex]).removeClass("active");

        $(progressBars[barIndex]).val("0");

        if (action == "next") {
            stepIndex++;
            barIndex++;
        }

        else if (action == "back") {
            stepIndex--;
            barIndex--;
        }

        $(steps[stepIndex]).addClass("active");
        $(progressBars[barIndex]).val("1");
    }


    // ALERT CHECK
    if ($(".alert").length) { // check if exists, no need for > 0 since 0 is false
        const flash = $('.alert').first();

        setTimeout(function() {
            flash.fadeOut(500);
        }, 2500)
    }

    // SET PROGRESS VALUES UNDER SLIDERS
    $('.slider-group').each(function() {
        let minVal = parseInt($(".slider", this).attr("min"));  // same as $(this).find('.min-desc').val()
        let maxVal = parseInt($(".slider", this).attr("max"));
        
        console.log(maxVal);
        if (maxVal <= 10) { // dont list too many numbers
            // properly adjust range
            if (minVal === 0) {
                maxVal++;
            }

            let progressValues = $(".progress-values", this);

            // Niko Ruotsalainen from https://stackoverflow.com/a/33352604/8341844
            // Equivalent of range() in Python with start/stop
            // Modified to create array with ints from minimum to maximum

            let progressRange = Array.from({length: (maxVal)}, (_, i) => i + minVal);

            $.each(progressRange, function(value) {
                $(progressValues).append(`<p>${value}</p>`);
            })
        }

        if (maxVal > 10) { // will display too many items
            $(progressValues).append(`<p>${minVal}</p>`);
            $(progressValues).append(`<p>${maxVal}</p>`);
        }

    })

});