# Pytorch Implement of Hierarchical Attention Networks 

This project reproduces the Hierarchical Attention Networks proposed in NAACL 2016 "Hierarchical Attention Networks for Document Classification". 

Here is the link of the original paper(https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

## Overall of Model Structure

-   The structure of proposed model is shown as follow: 
    -   First Use word embedding layer to get the word sequence.
    -   Use Bi-GRU model to get contextual representation of the word.
    -   Use $u_w$ (a learnable model parameters served as attention Query.) & Attention model aggregate word information into sentences representation.
    -   Similarly Use Bi-GRU model to get contextual representation of the sentence. Then Use $u_s$ (a learnable model parameters served as attention Query.) & Attention model aggregate sentence information into document representation.
    -   Finally Use a linear layer served as as classifier.

<img src="./asset/model.png" style="zoom:50%;" />

