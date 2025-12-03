# Co-segmentation

**Co-segmentation** discovers shared visual patterns across multiple images by processing them together as a unified graph. Instead of analyzing images independently, co-segmentation connects all images into a single large-scale affinity matrix, enabling the algorithm to find consistent semantic structures that span the entire image collection.

<div style="text-align: center;">
    <img src="../images/alignedcut_vs_ncut/affinity_aligned.jpg" style="width:75%;">
    <p><i>Unified affinity matrix connecting all images for co-segmentation</i></p>
</div>

<div style="text-align: center;">
    <img src="../images/alignedcut_vs_ncut/affinity_not_aligned.jpg" style="width:50%;">
    <p><i>Independent processing: multiple small affinity matrices</i></p>
</div>

## Why Co-segmentation?

When analyzing multiple related images, co-segmentation provides several key advantages:

- **Consistency**: Semantically similar regions receive consistent representations across all images
- **Better Pattern Discovery**: Discovers patterns that might be ambiguous in individual images but clear across the collection
- **Cross-Image Correspondence**: Enables finding matching regions across different images
- **Robust Segmentation**: Leverages information from multiple views to improve segmentation quality

The computational challenge of processing large unified graphs is solved through Nyström approximation, making co-segmentation scalable (see [How NCUT Works](../../methods/02a_nystrom_ncut_complexity)).

## Discovering Consistent Patterns

Co-segmentation ensures that semantically similar regions are represented consistently across images. In the example below, facial features and objects maintain the same color coding throughout the image collection, revealing the underlying visual structure.

<div style="text-align: center;">
<p><b>With Co-segmentation</b>: Consistent color representation across images. 
    <img src="../images/alignedcut_vs_ncut/face_aligned.jpg" style="width:100%;">
</p>
</div>

<div style="text-align: center;">
<p><b>Without Co-segmentation</b>: Each image processed independently with inconsistent representations.
    <img src="../images/alignedcut_vs_ncut/face_not_aligned.jpg" style="width:100%;">
</p>
</div>

## Cross-Image Correspondence

A powerful application of co-segmentation is finding corresponding regions across images. Since the eigenvector representations are aligned, we can measure similarity between any pixel and all other pixels across the entire image collection using simple distance metrics.
