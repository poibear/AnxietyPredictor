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

    function changeProgressVal(slider, progressDisplay) {
        let currentVal = parseInt($(slider).val());
        $(progressDisplay).find("p").text(currentVal);
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
        let slider = $(".slider", this);
        let progressValues = $(".progress-values", this);

        // Initialize first time progress value
        changeProgressVal(slider, progressValues);
        
        $(slider).on("input", function() {
            changeProgressVal(slider, progressValues);
        })
    })
});