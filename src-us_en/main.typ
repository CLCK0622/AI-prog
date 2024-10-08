#import "@preview/charged-ieee:0.1.0": ieee
#import "@preview/lovelace:0.3.0": *

#show: ieee.with(
  title: [Enhanced U-Net Usage on Road Network Prediction],
  abstract: [
    While infrastructural development is still growing tremendously, immediate concern is on a responsive road network prediction framework for quick-evolving environmental and social demands. In the conventional road planning methodology, the heavy reliance on time-consuming and labor-intensive manual surveys and data analysis in urbanization processes that are fast-growing has made it increasingly inefficient. After extensive exploration, this paper presents a very innovative application based on the architecture of U-Net, which utilizes geospatial data to predict road networks that could actually exist. In training the model, river networks, residential area data, and elevation data were used. Using Focal Loss with a `pos_weight` tensor, the model will care more about smaller roads and remote residential areas than before; these may be underrepresented when considering complex urban and rural landscapes due to class imbalance. Results have shown that while the model is limited with the data sources and computational power of the day and hence predicts large roads relatively well, it still demonstrates the full potential an AI model has in road network development and expansion. It improves regional connectivity that can help boost economic growth as well as contribute toward strategic smart city planning. This study represents significant development in road network prediction technology and especially points out its importance in improving the economic and social infrastructure by optimized road network construction.
  ],
  authors: (
    (
      name: "Yi (Kevin) Zhong",
      organization: [No. 2 High School of East China Normal University],
      location: [Shanghai, P.R.China],
      email: "zhongyi070622@gmail.com"
    ),
  ),
  index-terms: ("Road Network Prediction", "Deep Learning", "Urban Planning", "Focal Loss Optimization", "U-Net Architecture"),
  bibliography: bibliography("refs.bib"),
)

= Introduction

The pressing demands for infrastructure development call for immediate attention to a road network construction and expansion prediction framework that would adapt to changing environmental and social conditions. Traditional methods of road planning are heavily reliant on time-consuming and labor-intensive manual surveys and data analyses. However, with increasing rates of urbanization, these methods are becoming increasingly inefficient. It has, therefore, been very important to integrate geospatial data with computer technologies. Recent developments in remote sensing technologies and the widespread adoption of online maps have led to the unprecedented availability of rich geospatial datasets, including RS height data and roads and water data from platforms provided by governments and organizations. These datasets provide a solid foundation for utilizing deep learning to transform urban planning methods in the modern era.

The invention of ever more sophisticated machine learning models, increased computational power, and the development of various and extensive training datasets all combined to make deep learning applications both revolutionary and diverse.

In geospatial analysis, deep learning has played a vital role in solving complex prediction tasks such as recognizing traffic networks and predicting the flow of traffic @AI-road-network-prediction @deep-learning-road-traffic-prediction. These developments give strong evidence for the feasibility of deep learning in geospatial prediction, hence making machine learning-powered road planning possible.

As a result, this paper utilizes the U-Net architecture, which is demonstrated very well in image segmentation tasks. Further, it trains the model by means of the AdamW optimizer with IoU Loss and Focal Loss to predict the probable road network construction plan. The ultimate goal of this research is the development of deep learning techniques for improving accuracy and efficiency in road network prediction, showing how these technological advancements can be put into effective practice to support economic growth and strategic urban planning.

This paper is structured as follows: Section 2 presents a review of the related work concerning artificial intelligence technologies, geospatial data analysis, and road network prediction; Section 3 describes our experimental environment, including data processing, model configuration, and the training process; Section 4 reports and interprets the experimental results; Section 5 discusses the meaning of the prediction results, current limitations, and future directions; finally, Section 6 summarizes the paper and its contributions to the field, pointing out possible directions for future research.

= Related Work

The emergence of deep learning has significantly enhanced the ability to predict and analyze road networks and traffic flow using geospatial data. This section reviews recent advancements in this domain, highlighting methodologies, applications, and outcomes relevant to the scope of the current study.

== Deep Learning Basics

=== Convolutional Neural Networks (CNNs)

Convolutional Neural Networks form the cornerstone of modern image analysis, particularly in applications involving grid-like data structures such as images. Originating from the foundational work by LeCun et al. (1998), CNNs utilize layers designed for feature extraction and classification---convolutional layers, pooling layers, and fully connected layers @CNN. This architecture makes them exceptionally suited for image and video recognition tasks, including the semantic segmentation required for accurately detecting road networks from aerial imagery. @Comparison_CNN shows two typical CNN structures.

#figure(
  image("Comparison_CNN.svg"), 
  caption: [Two typical kinds of CNN: LeNet and AlexNet @CNNComparisonImage]
) <Comparison_CNN>

=== U-Net Architecture

Building on the capabilities of CNNs, the U-Net architecture introduced by Ronneberger, Fischer, and Brox (2015) specifically targets image segmentation challenges @U-Net. As shown in @Standard_U-Net , the "U-shaped" design of this architecture features a contracting path to capture context and a symmetric expanding path that enables precise localization. Though originally designed for medical uses, this structure is particularly effective for geospatial analysis tasks like road network prediction, where accurate segmentation of detailed features like roads from complex backgrounds is crucial.

#figure(
  image("u-net-illustration-correct-scale2.svg"),
  caption: [A standard U-Net structure @U-Net]
) <Standard_U-Net>

=== AdamW Optimizer

The optimization of deep learning architectures, such as U-Net, is critically enhanced by the AdamW optimizer. Developed by Loshchilov and Hutter (2017), AdamW improves upon the traditional Adam optimizer by decoupling weight decay from the optimization steps @AdamW. This adjustment addresses convergence issues found in the original Adam algorithm and leads to better generalization in training deep learning models. AdamW is particularly valuable in the training of neural networks for road prediction, where robustness and consistency across varied data are essential.

=== Focal Loss

To effectively train neural networks on imbalanced datasets, Focal Loss was introduced by Lin et al. (2017) @Focal-Loss. This advanced loss function is engineered to enhance the model’s focus on hard-to-classify examples by adjusting the standard cross-entropy loss based on the certainty of classification. In the context of road network prediction, where the presence of roads significantly underrepresents the geographical area, Focal Loss is instrumental in enhancing the model's detection capabilities, ensuring that minor but important features such as some remote residential spots are not overlooked.

== Applications

=== Integration of Multi-modal Data for Road Prediction

The study presented in “AI Powered Road Network Prediction with Multi-Modal Data” explores the efficacy of various data fusion methods in the context of road network detection @AI-road-network-prediction. The researchers utilized satellite imagery combined with GPS trajectory data, applying different fusion techniques such as early and late fusion. Findings indicated that early fusion, particularly through concatenation, yielded better Intersection over Union (IoU) metrics than late fusion, underscoring the potential of integrated approaches for enhancing road network predictions. This approach is particularly relevant to our methodology as it confirms the benefit of multi-modal data integration in improving prediction accuracy.

=== Deep Learning Frameworks for Traffic Prediction

In “A Deep Learning-Based Framework for Road Traffic Prediction,” a novel deep learning model combining convolutional neural networks (CNNs) and recurrent neural networks (RNNs) is proposed to improve traffic forecasting accuracy @deep-learning-road-traffic-prediction. The framework processes spatial data through CNNs and temporal dynamics through RNNs, tailored to urban traffic patterns. This dual approach aligns with our model’s architecture, where spatial and temporal features are crucial for accurate road prediction.

=== Conversion of Geospatial Data for Predictive Analysis

The transformation of geospatial data into image-like formats to harness deep learning algorithms is discussed in “Geospatial data to images: A deep-learning framework for traffic prediction” @geospatial-data-to-images. This study addresses the preprocessing necessary to adapt traditional geospatial data for compatibility with neural network architectures, which is directly applicable to the preprocessing steps employed in our research.

=== Big Data Approaches to Traffic Flow Prediction

Lastly, the “Traffic Flow Prediction with Big Data: A Deep Learning Approach” research illustrates how deep learning can process extensive datasets to forecast traffic conditions effectively @traffic-flow-prediction. By implementing a deep neural network architecture capable of handling vast amounts of data, this study provides insights into scaling and managing large-scale data inputs, a crucial aspect of our work where data volume can significantly impact predictive performance.

These studies collectively demonstrate the robustness of deep learning techniques in processing and predicting complex patterns within road and traffic datasets. Our research builds upon these foundations, aiming to refine and extend the applications of these technologies to not only predict existing road patterns but also suggest potential areas for network expansion to enhance connectivity and economic growth.

= Data and Techniques

== Data Processing

Given that we are utilizing an advanced deep learning model on this topic, the selection and application of data is important. In this section, I will introduce several main data collection method and ways to execute the data. 

=== Data Source

Data for this project is mainly from several institutions: the National Geomatics Center of China, the National Catalogue Service For Geographic Information, the National Earth System Science Data Center, and the Geospatial Data Cloud. According to some geographic studies(ref), we should consider a large variety of factors, including height, slope, soil, climate, and residential area, when designing a new road. Specifically, the evaluation of a road design should focus on lowering the cost and stimulating the connection between areas. However, because of the limitation of data source and computing power, we simply choose height(`.tiff`), rivers(`.shp`), and residential areas(`.shp`) as input, while the road network as output.

=== Clipping and Re-projection

For better efficiency and lower computing power, we clip the data into $1500 times 1500$ pixels. As these datasets are from different sources and have different coordinate systems, we can read and reproject river, residential, road, and boundary data into EPSG:4490 with `geopandas` @epsg4490.

=== Visualization

We choose Ningxia Hui Autonomous Region as an example. The following graph is a picture generated from the geospatial data mentioned before. We utilize `matplotlib` for this step.

According to @Ningxia-geospatial-data, the blue lines are rivers, while the red ones refer to roads. These grey spots are residential spots.

#figure(
  image("GPD-Ningxia.png"),
  caption: [Geospatial data in Ningxia, China as an example.]
) <Ningxia-geospatial-data>

After re-projecting coordinate systems and combining all single height files together, we can directly visualize the height data with `rasterio`, `geopandas` and `matplotlib`. To save resources as mentioned above, we clipped the height data to $1500 times 1500$ pixels. After visualization, we normalized the height data for better training.

#figure(
  image("clipped_DEM.png"),
  caption: [Height data (clipped & normalized) in Ningxia, China.]
)

=== Re-organized Data Structure

To utilize these data together as an input source, we can combine all data above into the DEM data, which is $2000 times 2000$ pixels. In this way, the input source will have 3 channels, and output should be a single channel.

After that, the data has to be split into three data-loaders. Specifically, 60% is for training, 20% for validation, and the rest for testing. 

== Framework

Originally developed for medical uses, the U-Net architecture is well-suited for processing image masks. By integrating it with other deep learning techniques, we can adapt this powerful framework to effectively predict road networks, given its strong image segmentation capabilities and efficient structure.

=== Model Architecture

We utilize the U-Net architecture, a convolutional neural network originally designed for biomedical image segmentation. This kind of architecture is highly effective in situations which need accurate location because it has two features, a contraction part for the context and an expansion part to enable exact localization. By the implementation of several double convolutional blocks that consist of two sets of convolutions with interjections of batch normalization and ReLU, the U-Net architecture is important for high performance in road segmentation as it helps to capture such intricate features that come in various sizes, from different geospatial data such as satellite images and elevation maps. The architecture consists of several down-sampling layers which allow the amount of features to increase while decreasing the spatial size of the data, then one or several up-sampling layers are applied to bring back the size of the features and interlace the features from the corresponding down-sampling layers via skip mechanisms. This allows finer visual detail, which is usually lost, to be retained during the down-sampling procedure which results in the model recovering most vital parts of the road across different terrains.

#figure(
    kind: "algorithm",
    supplement: [Algorithm],
    caption: [Enhanced U-Net Architecture],
    pseudocode-list(booktabs: true, stroke: none)[
        - *Define* U-Net Architecture:
          - Input Channels = 3
          - Output Channels = 1 (Binary mask)
          - *Layers*:
            + DoubleConv ($text("Input") -> 64 text("channels")$)
            + DownSampling ($64 -> 128$, $128 -> 256$, $256 -> 512$, $512 -> 512$)
            + UpSampling with skip connections ($512 -> 256 -> 128 -> 64$)
            + Output Convolution ($64 -> 1$, Sigmoid Activation)
    ]
)

=== Loss Functions

#figure(
    kind: "algorithm",
    supplement: [Algorithm],
    caption: [Loss Functions],
    pseudocode-list(booktabs: true, stroke: none)[
        - *Define* FocalLoss:
          - *Parameters*: $alpha=0.5$, $gamma=3$
          - *Functionality*:
            - Compute BCE Loss with logits
            - Adjust loss focus based on ground truth relevance and prediction probability
        
        - *Define* ConnectivityLoss:
          - *Parameters*: $alpha=0.001$
          - *Functionality*:
            - Use a $3 times 3$ convolution kernel to dilate predicted masks
            - Compute the mean difference between dilated mask and original prediction
    ]
)

In order to solve problems of class imbalance and the difficulty of predicting a road network, our model employs two specialized loss functions: Focal Loss and a custom Connectivity Loss. Focal Loss is designed to fine-tune the model's focus towards hard-to-classify instances, in this case enhancing the performance of the model on minority classes by altering this normal cross-entropy loss to include a modulating factor. This helps to bring down the weight of the losses for well-classified instances, such that the model concentrates on difficult examples, which are common in circumstances where roads cover a small fraction of the geographical field. In this case, we use an $text(alpha)$-balanced variant of the focal loss:

$ "FL"(p_t) = -alpha_t (1-p_t)^gamma log(p_t). $

The Connectivity Loss further augments the model's capability by encouraging spatial contiguity in the predicted road networks. It assigns a penalty to the model for not predicting connected stretches of roads thus enhancing the chances of predicting interconnected roads. Hence, the consequent optimal output coming from this combination of loss functions is a road map that is probably more usable in the real world for urban design and traffic optimization.

#figure(
    kind: "algorithm",
    supplement: [Algorithm], 
    caption: [Connectivity Loss],
    pseudocode-list(booktabs: true, stroke: none)[
      - *Function* connectivity_loss(mask, alpha=0.001):
        - *Description*:
          - Calculate loss to encourage spatial connectivity in predicted road masks.
        - *Steps*:
          + Create a 3x3 kernel filled with ones.
          + Apply dilation using convolution with padding to maintain dimensions.
             + dilated = conv2d(mask, kernel, padding=1)
          + Convert dilated output to binary (values > 0 set to 1).
          + Calculate the difference between the dilated mask and original mask.
          + Compute the mean of the difference and scale by alpha.
             + $"CL" = "mean"("dilated" - "mask") times alpha$
        - *Return*:
          - connectivity_loss: Scalar value representing the loss due to disconnected areas.
    ]
)

=== Training Process

The training process is meticulously designed to optimize the U-Net model over multiple epochs using the AdamW optimizer, a variant of the Adam optimizer that incorporates weight decay to combat overfitting. The model is trained using a dataset split into 60% training, 20% validation, and 20% testing portions, ensuring comprehensive coverage and robust testing across diverse geospatial features. Each training session involves forward propagation of batches of input data through the network, followed by the computation of losses using the aforementioned Focal and Connectivity Loss functions. Backpropagation adjusts the model weights to minimize these losses, with the AdamW optimizer facilitating efficient convergence by adjusting learning rates based on weight decay and moment estimates. Validation runs intermittently provide feedback on model performance, preventing overfitting and ensuring the model's generalizability to unseen data. The training loop is enhanced with connectivity loss adjustments that progressively increase its influence, ensuring that the model increasingly prioritizes the prediction of connected road networks as training progresses. This structured training regimen ensures that the model not only learns to predict road presence but does so in a way that is practically applicable for effective road network analysis and planning.

#figure(
    kind: "algorithm",
    supplement: [Algorithm],
    caption: [Training Process],
    pseudocode-list(booktabs: true, stroke: none)[
      + *For* each epoch:
        + Perform forward pass with U-Net on training data
        + Calculate Focal Loss and Connectivity Loss
        + Backpropagate total loss
        + Update model weights using AdamW optimizer
        + Validate model performance on the validation set
        + Record training and validation losses
    ]
)

= Results

= Discussion

= Conclusion
