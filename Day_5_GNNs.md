# Day 5
## Graph Neural Networks
My notes from youtube video by [PyData](https://www.youtube.com/watch?v=b187J4ndZWY)

![Timeline of AI](https://github.com/tanyajainC137/100-Days-of-ML/blob/master/AI%20timeline.JPG)

#### Benefits of GNNs
1. Best for handling non-euclidian data like graphs and Point Clouds
1. Reduced false positives and false negatives
2. Better at handling adversial noise (targeted minor change to inputs to confuse the AI)
3. Better at exploiting the usefulness of locality (neighbours) and weight-sharing (associated with vertices)
4. GCNs make the images in a the Convolutional Network as graphs by inserting **Peer Regularizations** in between the layers. These graphs hold representative significance of the images. These collection of graphs (output imgs) themselves form a graph altogether.

#### Applications
* In medicine and drug discovery; early detection of diseases; brain function research; molecular chemistry.
* Recommender system enhancement; fake news detection; self-driving cars' vision advancement

#### Working of GCNs

![Graph convolution](https://github.com/tanyajainC137/100-Days-of-ML/blob/master/GNN%20math.JPG)

In the image above, 
        `# Graph convolution,    X'i = Agg Op over j  x  edge func `
After the summation and aggregation
        `#The function reduces to,     X' = A(Th)  x  (X)     where A(Th) is a nonlinear transform on the inputs X`
        
