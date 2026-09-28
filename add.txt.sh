
rm -vf text.padded.cleaned.aligned.png
rm -vf padded.cleaned.aligned.png

magick ./thumbnail_cleaned/059.tif -crop 1696x728+150+50 +repage ./thumbnail_cleaned/059.png
magick ./thumbnail_cleaned/060.tif -crop 1696x728+150+50 +repage ./thumbnail_cleaned/060.png
magick ./thumbnail_cleaned/061.tif -crop 1696x728+150+50 +repage ./thumbnail_cleaned/061.png

magick ./thumbnail_aligned/059.tif -crop 1696x728+150+50 +repage ./thumbnail_aligned/059.png
magick ./thumbnail_aligned/060.tif -crop 1696x728+150+50 +repage ./thumbnail_aligned/060.png
magick ./thumbnail_aligned/061.tif -crop 1696x728+150+50 +repage ./thumbnail_aligned/061.png


magick ./thumbnail_cleaned/059.png ./thumbnail_cleaned/060.png ./thumbnail_cleaned/061.png -background white +smush 20 top.png
magick ./thumbnail_aligned/059.png ./thumbnail_aligned/060.png ./thumbnail_aligned/061.png -background white +smush 20 bottom.png 
magick top.png -size 1x20 canvas:white bottom.png -append cleaned.aligned.png
magick cleaned.aligned.png -background white -gravity east -extent 5500x1500 padded.cleaned.aligned.png
			


magick padded.cleaned.aligned.png \
-font Helvetica-Bold -pointsize 48 -fill black -annotate +20+130 "Unaligned sections" \
-font Helvetica-Bold -pointsize 48 -fill black -annotate +20+900 "Aligned sections" \
-font Helvetica-Bold -pointsize 48 -fill white -annotate +550+130 "Section 59" \
-font Helvetica-Bold -pointsize 48 -fill white -annotate +550+900 "Section 59" \
-font Helvetica-Bold -pointsize 48 -fill white -annotate +2250+130 "Section 60" \
-font Helvetica-Bold -pointsize 48 -fill white -annotate +2250+900 "Section 60" \
-font Helvetica-Bold -pointsize 48 -fill white -annotate +3875+130 "Section 61" \
-font Helvetica-Bold -pointsize 48 -fill white -annotate +3875+900 "Section 61" \
text.padded.cleaned.aligned.png


rm -vf ./thumbnail_cleaned/*.png
rm -vf ./thumbnail_aligned/*.png
