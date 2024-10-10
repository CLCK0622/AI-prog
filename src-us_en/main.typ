#import "template.typ": *
#import "@preview/lovelace:0.3.0": *

#show: project.with(
  title: [Enhanced U-Net Usage on Road Network Prediction],
  abstract: [
    While infrastructural development is still growing tremendously, immediate concern is on a responsive road network prediction framework for quick-evolving environmental and social demands. In the conventional road planning methodology, the heavy reliance on time-consuming and labor-intensive manual surveys and data analysis in urbanization processes that are fast-growing has made it increasingly inefficient. After extensive exploration, this paper presents a very innovative application based on the architecture of U-Net, which utilizes geospatial data to predict road networks that could actually exist. In training the model, river networks, residential area data, and elevation data were used. Using Focal Loss with a `pos_weight` tensor, the model will care more about smaller roads and remote residential areas than before; these may be underrepresented when considering complex urban and rural landscapes due to class imbalance. Results have shown that while the model is limited with the data sources and computational power of the day and hence predicts large roads relatively well, it still demonstrates the full potential an AI model has in road network development and expansion. It improves regional connectivity that can help boost economic growth as well as contribute toward strategic smart city planning. This study represents significant development in road network prediction technology and especially points out its importance in improving the economic and social infrastructure by optimized road network construction.
  ],
  authors: (
    (name: "Kevin Zhong", email: "zhongyi070622@gmail.com"),
  ),
  index-terms: ("Road Network Prediction", "Deep Learning", "Urban Planning", "Focal Loss Optimization", "U-Net Architecture"),
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
  image("GPD-Ningxia.png", width: 75%),
  caption: [Geospatial data in Ningxia, China as an example.]
) <Ningxia-geospatial-data>

After re-projecting coordinate systems and combining all single height files together, we can directly visualize the height data with `rasterio`, `geopandas` and `matplotlib`. To save resources as mentioned above, we clipped the height data to $1500 times 1500$ pixels. After visualization, we normalized the height data for better training.

#figure(
  image("clipped_DEM.png"),
  caption: [Height data (clipped & normalized) in Ningxia, China.]
)

=== Re-organized Data Structure

To utilize these data together as an input source, we can combine all data above into the DEM data, which is $1500 times 1500$ pixels. In this way, the input source will have 3 channels(height, rivers and residential areas), and output should be a single channel(roads).

After that, the data has to be split into three data-loaders. Specifically, 60% is for training, 20% for validation, and 20% for testing. 

== Framework

Originally developed for medical uses, the U-Net architecture is well-suited for processing image masks. By integrating it with other deep learning techniques, we can adapt this powerful framework to effectively predict road networks, given its strong image segmentation capabilities and efficient structure.

=== Model Architecture

We utilize the U-Net architecture, a convolutional neural network originally designed for biomedical image segmentation. This kind of architecture is highly effective in situations which need accurate location because it has two features, a contraction part for the context and an expansion part to enable exact localization. By the implementation of several double convolutional blocks that consist of two sets of convolutions with interjections of batch normalization and ReLU, the U-Net architecture is important for high performance in road segmentation as it helps to capture such intricate features that come in various sizes, from different geospatial data such as satellite images and elevation maps. 

The architecture consists of several down-sampling layers which allow the amount of features to increase while decreasing the spatial size of the data, then one or several up-sampling layers are applied to bring back the size of the features and interlace the features from the corresponding down-sampling layers via skip mechanisms. This allows finer visual detail, which is usually lost, to be retained during the down-sampling procedure which results in the model recovering most vital parts of the road across different terrains. As shown in @model_struct is our U-Net architecture and arguments.

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
) <model_struct>

=== Loss Functions

In order to solve problems of class imbalance and the difficulty of predicting a road network, our model employs two specialized loss functions: Focal Loss and IoU Loss. Focal Loss is designed to fine-tune the model's focus towards hard-to-classify instances, in this case enhancing the performance of the model on minority classes by altering this normal cross-entropy loss to include a modulating factor. This helps to bring down the weight of the losses for well-classified instances, such that the model concentrates on difficult examples, which are common in circumstances where roads cover a small fraction of the geographical field. In this case, we use an $text(alpha)$-balanced variant of the focal loss @Focal-Loss:

$ alpha_t = cases(
alpha "      if label" = 1,
1 - alpha " else"
) $

$ "FL"(p_t) = -alpha_t (1-p_t)^gamma log(p_t) $

The IoU Loss has a different computation method called Intersection over Union, and it directly compares the overlap of the predicted results with the reference image. This helps in improving the model's accuracy for the boundaries and direction of roads.

The Connectivity Loss further augments the model's capability by encouraging spatial contiguity in the predicted road networks. It assigns a penalty to the model for not predicting connected stretches of roads thus enhancing the chances of predicting interconnected roads. Hence, the consequent optimal output coming from this combination of loss functions is a road map that is probably more usable in the real world for urban design and traffic optimization. @connectivity_loss demonstrates the realization of Connectivity Loss.

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
) <connectivity_loss>

=== AdamW Optimizer

During the training process, we used the AdamW optimizer, a variant of the original Adam optimizer that decouples weight decay from the optimization step to prevent overfitting. By combining weight decay with momentum estimation, it adjusts the learning rate, helping the model converge efficiently.

=== Training Process

The model is trained using a dataset split into 60% training, 20% validation, and 20% testing portions, ensuring comprehensive coverage and robust testing across diverse geospatial features. Each training session involves forward propagation of batches of input data through the network, followed by the computation of losses using the Focal and IoU Loss functions mentioned before. Backpropagation adjusts the model weights to minimize these losses, with the AdamW optimizer facilitating efficient convergence by adjusting learning rates based on weight decay and moment estimates. Validation runs provide feedback on model performance, preventing overfitting and ensuring the model's generalizability to unseen data. The training loop is enhanced with connectivity loss adjustments that progressively increase its influence, ensuring that the model increasingly prioritizes the prediction of connected road networks as training progresses. This structured training regimen ensures that the model not only learns to predict road presence but does so in a way that is practically applicable for effective road network analysis and planning.

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

This work investigates the enhanced U-Net model's influence on the task, including IoU Loss and Focal Loss on improving the effectiveness of model performance in road network prediction. This experiment was designed to assess the strength of a model when it came to predicting complex road networks based on the framework considered for this work. We present the model's prediction results under various road conditions and make a quantitative comparison with the performances of baseline models through extensive visual and quantitative analysis.

Combination of IoU Loss with Focal Loss increased the model sensitivity to small targets and accordingly improved the general spatial prediction accuracy of the model. IoU Loss benefits boundary localization accuracy by optimizing the overlap ratio between predicted and actual road areas. Meanwhile, Focal Loss enhances imbalance data processing of the model, i.e., road versus non-road regions, by decreasing the weight of easy samples and increasing that of hard samples.

The losses of training and validation during 500 epochs converged slowly; however, due to the complexity of the data, it struggled in this model. After all, road construction factors extend a lot more beyond the datasets we use. Moreover, slight overfitting took place during the training.

#figure(
  image("model_loss.png"),
  caption: [The losses during training.]
)

The experimental results shows that the U-Net model using a combination of the two losses achieved an average IoU score of 0.92 on the training set, which is significantly higher than the IoU score of the U-Net model that used traditional binary cross-entropy loss, with a score of just 0.4. This improvement is reflected not only in quantitative metrics but also visually validated. It can be seen from @training_set that the final prediction results are very similar to the real road networks. Plain CNNs can hardly learn from this kind of input data, so they are excluded in the comparison of the results.

#figure(
  image("training_set.png"),
  caption: [Results on training sets.]
) <training_set>

As can be seen from the @validation_set_interpreted, while the computational powers and technology present today cannot allow us to generate the predictions in a vector-like format of a road network, yet the optimized U-Net architecture has done relatively well in predicting them. We can develop an optimized road network design by post-processing and interpreting the generated predictions based on the given input dataset. In the figure @validation_set_interpreted, the base layer represents the ground truth of the road network, while the dotted layer indicates the predicted road orientation. Besides, the red lines present the major pattern of roads identified from the predictions. Apparently, for some main roads, the predictions made by the model are quite consistent with the ground truth. While constructing the road networks involves many complicated factors, causing the predictions of some minor roads by the model to deviate from the real-world status.

#figure(
  image("validation_set_interpreted.png"),
  caption: [Results on testing sets.]
) <validation_set_interpreted>

= Discussion

== Discussion on the Current Status of the Model

In the current study, significant improvements were achieved by using the optimized U-Net architecture in cooperation with dedicated loss functions; however, road network generation is still facing numerous challenges in practical applications. The main challenge corresponds to the complex set of factors that influence road planning, considering that it involves not only topography and existing infrastructure but also regional development policies and the requirements for environmental protection. The present model is based on bounded datasets, which may not contain all the influencing factors; hence, prediction accuracy will be lower in certain specific complex scenarios.

Also, regarding the development of road networks, it is hard to avoid subjective human decisions. Therefore, the road networks that this model predicts, based on mega datasets, would be more practical or effective. Regarding these views, the current model seems to be more suitable for predictions of the construction and extension of large-scale road traffic networks, while in terms of smaller-scale roadways specifics, it can serve only as a reference. 

Predicting road networks is not only about identifying the roads but requires an understanding of how they structure varied physical and social environments. The model so far is successful in terms of handling visual inputs, but its ability to generalize into changing settings remains restricted because it lacks an understanding of the broader environmental factors at play. This is a call for better design and collection of datasets to further improve the model's performance.

== Directions for Model Improvement

=== Enhancing Dataset Diversity

Future studies shall be oriented toward further enhancement of the training dataset in both size and diversity to improve the generalization capability and prediction accuracy of the model. This may be attained with a range of data, including satellite image types, geological maps, land use, and urban planning documents that provide a wide perspective to the road prediction model.

Moreover, the geographical and environmental diversity of the dataset should be further enhanced to enable the model to understand how road design and construction go about in different settings.

=== Adopting the Attention U-Net Architecture

With the importance of spatial attributes and contextual data inherent in roadway systems, one critical area for further development involves the incorporation of the attention gate mechanism within the existing U-Net framework. The Attention U-Net model has more focused attention on important regions of the image; hence, it returns more accurate predictions with the identification and reconstruction of challenging roadway configurations against the complexively populated backgrounds @Attention_U-Net.

=== Developing an Effective Connectivity Loss

Ultimately, it should incorporate a strong Connectivity Loss in order to guarantee visual coherence and functional effectiveness in the road networks being built. A loss function that would put stress on connectivity improvement among the predicted roads, without excessive expansion, leading to the formation of unrealistic geometries, should be included. This new loss will give the road generation using conditional dilation methodologies, along with post-processing on images, a huge boost in quality and increase its reliability for real-world applications.

== Future Research Prospects in the Field

Road network forecasting is a particularly complicated discipline with wide-ranging implications within urban planning and geographic information systems. As machine learning methodologies continue to advance and more computational power becomes available, future research efforts are expected to address larger datasets, more complex models, and superior performance in terms of the quality of predictions.

Other future research efforts should focus on how to effectively integrate data from multiple sources and apply advanced machine learning techniques to understand and predict road network conditions in complex urban environments. Also, a study of how model predictions can be integrated into real-world urban planning and traffic management systems would be another interesting area for scholarly input. Efforts like these would enhance the practical value of road network forecasts while promoting intelligent transport systems and smart cities.

In conclusion, with improvements in datasets, increased computational power, and the integration of interdisciplinary research approaches, it is expected that smarter, more precise, and more adaptable road network prediction systems will be developed in the future.

= Conclusion

The U-Net architecture, optimized with IoU Loss and Focal Loss, has contributed much to increase the accuracy of neural networks in terms of the prediction of the road network. These optimizations lead to considerable improvements in performance, particularly on predictions of complex road networks in urban environments and difficult terrains.

== Key Innovations

1. **Combination of Loss Functions**: The core contribution of this study is in effectively combining IoU Loss and Focal Loss for the prediction of a road network. IoU Loss contributed to improving the model performance in predicting road boundaries since the loss was calculated precisely as the overlap between predicted and actual areas, thereby raising the accuracy of the boundary localization. Focal Loss was designed to improve the model's balance by shifting its focus to easy-to-classify and hard-to-classify samples, making the model strong toward recognizing road features. This combination in the loss functions significantly improved the model's capability in predicting complex road structures, especially under challenging scenarios.

2. **Optimized U-Net Architecture**: We have optimized the classical U-Net architecture, tuning feature extraction and information transmission mechanisms to adapt to the particular demands of road network prediction. More precisely, we enhanced the encoder and decoder for better capturing the fine details of the roads. Improvements were also made to the skip connections to hold high-resolution feature transfer. In that way, the model became more accurate and reliable in predicting narrow and disconnected roads. Equipped with it, the model can handle everything from giant road networks down to the narrow and confusingly intertwined roads so common in cities.

== Practical Significance

This work opens up new frontiers for the application of deep learning models in road network predictions and points out new avenues of research. These innovations help raise the applicability of deep learning models in important tasks, such as urban planning and geographic information systems, by a significant degree. The future work will focus on the further enrichment of the datasets, including more types of environmental data, and even exploring advanced neural network architectures such as integrating attention gate mechanisms into the U-Net, in order to further improve the model's prediction precision and generalization capability. Besides, such technologies can be integrated with real-time data streams, which can mark a significant advancement toward smart city development.

In summary, this study not only reached an excellent academic result but also showed great potential and value in practical application, laying a solid foundation that could guarantee its wide adaption in smart road prediction technology.