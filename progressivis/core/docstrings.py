# flake8: noqa

FILENAMES_DOC = (

            "Files to read. The underlying {{PTable}} must have a ``filename`` column containing the file URIs."
            " |:warning:| ``filenames`` slot and ``filepath_or_buffer`` init parameter  cannot both be defined!"

)


RESULT_DOC = "Provides read data into a {{PTable}} object"


INPUT_SEL = ("Data entry. A selection of columns to "
             "be processed could be provided via a hint"
             " (see :ref:`example <hint-reference-label>`)."
             " When no hint is provided all input columns are processed"
             )
