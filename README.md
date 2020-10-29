This code is no longer maintained, it serves as an archive of a project I did while I was at Lefty, when it was an image search engine on instagram. It is provided with permission from Lefty's CTO.

You can find a presentation of similia in [similar_images_pq.pdf](similar_images_pq.pdf). Note that after the presentation I wrote an optimization that reduced searching time to about 100ms, with 1B+ images in the database.

We used it to search visually similar images on social media without using keywords. It could be used for any nearest neighbor search in an euclidean space.

This work was inspired by the following papers:
- Product Quantization for Nearest Neighbor Search by Jégou et al https://hal.inria.fr/inria-00514462v2
- The Inverted Multi-Index by Babenko et al http://www.robots.ox.ac.uk/~vilem/cvpr2012.pdf
- Optimized Product Quantization for Approximate Nearest Neighbor Search by Ge et al http://kaiminghe.com/publications/cvpr13opq.pdf
- Improving Bilayer Product Quantization for Billion-Scale Approximate Nearest Neighbors in High Dimensions by Babenko et al https://arxiv.org/abs/1404.1831
