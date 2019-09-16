;
(function ($, window, document, undefined) {
    'use strict';
    $(document).ready(function ($) {
        $("#uploaded-file").on('change', function(){
            console.log("submitting form");
            $("#frm-file-uploader").submit();

        });
    });
}(jQuery, window, document));