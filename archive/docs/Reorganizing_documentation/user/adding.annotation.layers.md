## Adding annotion layers from different brains
1. Open up a blank page in neuroglancer at: https://activebrainatlas.ucsd.edu/ng_multi/
1. We need to load an image layer first from a neurotrace brain:
    1. In the right part of the page where it says "Data source URL" add an image layer. When you click on that tab, a drop down of available image loading types appears, select "precomputed" and paste this: https://activebrainatlas.ucsd.edu/data/DK52/neuroglancer_data/C1
    1. Hit the enter key twice to load it.
    1. Zoom out to get a good view in all 4 quadrants.
1. Add the atlas:
    1. Click the "+" sign next to the image you just loaded near the top left of the screen. Again, the list of loading types appears. Copy and paste this into the "precomputed" area. https://activebrainatlas.ucsd.edu/data/structures/atlasV7 and hit Enter twice.
    1. Click the "Seg." tab and. You should see a list of structures, click the checkbox next to the "28 listed ids".
1. The first image layer is not aligned. You can either align it or hide if you don't want to see it.
    1. To hide it, simply click the "X" in the top left rectangle that has the "C1" in it.
    1. To align it, click in the "C1" rectangle near the top left and then go to the "Source" tab on th right. Look near the bottom of the "Source" tab and find the dropdown for alignment. It should already say "DK52 manual beth". If not, find that selection in the dropdown and click the "Align" link. You'll need to move the brain into the center of the quadrants.

### Adding annotation layers
1. Click the "+" button in the top left next to the "atlasV7" rectangle. Make sure you use control+click as we are adding annotation layers.
    1. In the dropdown menu in the right side of the page, select an annotation layer, in this example, we'll try "DK39_Hannah_Ann/premotor". After selecting that, click the "Import" link next to it.
    1. Change the size of the dots by going to the "Rendering" tab. Increase the size, 3 is a good size. You should be able to see the dots in the lower left quadrant. They need to be aligned.
    1. Still on the new annotation layer you created, go to the "Source" tab at the top right of the page.
    1. Look for the "Select transformation" dropdown at the bottom right side. 
    1. Select "DK39 manual Beth" and click Align. The points should move into the brain stem area of the brain.
    1. Repeat the above steps with another annotation layer. Make sure you align the set of points with the same brain as the set of points belongs to.
