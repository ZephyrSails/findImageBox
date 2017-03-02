# findImageBox

This code will detect text area in flowchart, and applying OCR to read them. Modified based on [xiaofeng's work](https://github.com/ZephyrSails/ocrnn).

# USAGE:

test getting single box area:

    ~ python getBox.py flowchart_data_set/flowchart1.vsd_page_1.png

crop the text area: This will produce crop/ holding cropped text area from images in flowchart_data_set

    ~ python findTextAreas.py


apply OCR on cropped data:

    ~ python ocr.py
