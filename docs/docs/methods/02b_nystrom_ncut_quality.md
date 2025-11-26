# Nyström Ncut (Quality)

TODO: rewrite this document, it taste like AI

The Nyström NCut approximation not only improves computational complexity but also **enhances segmentation quality** compared to standard NCut approaches.

## Quality Improvement through FPS Sampling

The key quality improvement comes from **Farthest Point Sampling (FPS)**, which addresses a fundamental issue in graph-based segmentation: **class imbalance**.

### Reducing Class Imbalance

In typical datasets, certain regions or classes are over-represented while others are under-represented. 

**Toy 2D Example**: Consider a 2D point cloud with 1000 blue points clustered in one region and only 50 red points scattered in another region. With random sampling:
- 95% of sampled points would likely be blue
- The resulting eigenvectors would be heavily biased toward the blue cluster
- The red cluster boundaries would be poorly defined or completely missed
- NCut would tend to merge red points into the dominant blue cluster

This class imbalance can lead to:
- Biased eigenvectors that favor dominant classes
- Poor segmentation boundaries for minority classes
- Suboptimal clustering results

**FPS sampling creates a more balanced representation** by:
1. **Ensuring spatial diversity**: FPS selects nodes that are maximally separated in feature space
2. **Preventing clustering bias**: No single class or region dominates the sampled subset
3. **Maintaining representative coverage**: All important regions get proportional representation

This balanced sampling leads to eigenvectors that better capture the true structure of the data, resulting in higher quality segmentations across all classes, not just the dominant ones.

### Empirical Quality Benefits

The FPS-based Nyström approximation consistently produces:
- More balanced cluster assignments
- Better boundary preservation for minority classes  
- Improved segmentation consistency across different scales
<div style="display: flex; justify-content: space-between; gap: 20px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0;">
  <a href="/methods/02a_nystrom_ncut_complexity" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; transition: all 0.2s;">
    <span style="font-size: 12px; color: #666; margin-bottom: 5px;">Previous</span>
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">← Nyström Ncut (Complexity)</span>
  </a>
  <a href="/methods" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; transition: all 0.2s;">
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">Back to Overview</span>
  </a>
  <a href="/methods/03_kway_ncut" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: flex-end; text-align: right; transition: all 0.2s;">
    <span style="font-size: 12px; color: #666; margin-bottom: 5px;">Next</span>
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">K-way Discrete Ncut →</span>
  </a>
</div>
