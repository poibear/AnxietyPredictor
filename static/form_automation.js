// credits to WEB CIFAR @ https://www.youtube.com/watch?v=cKTgIDkRsGc for form & js
// using jquery instead of traditional js
$(document).ready(function () {
    const steps = $('.step');
    const nextBtn = $('.next-btn');
    const prevBtn = $('.back-btn');
    const form = $('.form');

    
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
        let index = 0; // start @ 0th step
        const active = $("form .step.active");
        index = steps.index(active);
        $(steps[index]).removeClass('active');

        if (action == "next") {
            index++;
        }

        else if (action == "back") {
            index--;
        }

        $(steps[index]).addClass("active");
        console.log(index);
    }


    if ($(".alert").length) { // check if exists, no need for > 0 since 0 is false
        const flash = $('.alert').first();

        setTimeout(function() {
            flash.fadeOut(500);
        }, 2500)
    }
});