# deepdream-pytorch

**Pytorch implementation of the DeepDream algorithm.**

It is not finished yet.


Not everyone has a large GPU so I liked the idea about processing high resolution images in tiles. 

There was some problems.
First of all, when I divided the image into tiles and fed each of them to the network input the gradient plot of the whole image was like this: 

![seams](/outputs/output_seams.png)

I think this is because pixels at edges make a smaller contribution to activations with "Valid" padding in the convolution. Hence in some models the gradient at the edges is smaller. To fix this problem I added padding to the tiles at the feeding stage and subsequent cropping. One can use zero-padding but when it's possible I just take extended tile of the original image. It gave me 

![intensity_difference](/outputs/output_std.png)

Then I slightly changed the normalization of the tile gradients in an assumption that we roughly want E(g1) = E(g2), where g1,g2 are gradients of two different tiles. Finally I got

![grad_map](/outputs/output_mean.png)
