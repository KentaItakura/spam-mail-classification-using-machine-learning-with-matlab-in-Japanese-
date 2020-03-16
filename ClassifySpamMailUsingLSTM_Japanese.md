# LSTM��p�������f���[���̌��o


���̗�ł́ALSTM (Long Short Term Memory) �ƌĂ΂���@��p���āA���̕��ʂ�����f���[�����ǂ����𔻕ʂ��܂��B




���̗�̂ق��Ƀi�C�[�u�x�C�Y�ƌĂ΂���@��p�������̂�����܂��B




�܂��A������̃R�[�h�̏������͉���matlab�����h�L�������g���Q�l�ɂ��܂����B




([https://jp.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html](https://jp.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html))


# �f�[�^�̃C���|�[�g


����p����f�[�^�́A������i[https://www.kaggle.com/uciml/sms-spam-collection-dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) )�ɂ���




SMS Spam Collection Dataset Collection of SMS messages tagged as spam or legitimate�@�Ƃ����f�[�^�Z�b�g�ł��B




���v�ŁA5574���̃��[��/�X�p�����[�����p�ӂ���Ă��܂��B���URL����f�[�^���_�E�����[�h����ƁA




spam.csv�Ƃ����t�@�C���𓾂邱�Ƃ��ł��܂��B�����ǂݍ��݂܂��B




�G�N�Z���Ƀ��x���₻��ɑΉ����镶�͂��L�^����Ă���ꍇ�́Areadtable�֐����g���ƕ֗��ł��B




�ϐ�����data�Ƃ��āA�G�N�Z���t�@�C���̏���ǂݍ��݂܂��B




head�֐��ɂēǂݍ��񂾃t�@�C���̓��e�̈ꕔ����y�Ɋm�F�ł��܂��Bv1��ɖ��f���[��(spam)�������łȂ���(ham)�������Ă��܂��B


```matlab
clear;clc;close all
filename = "spam.csv";
data = readtable(filename,'TextType','string');
head(data)
```
| |v1|v2|Var3|Var4|Var5|
|:--:|:--:|:--:|:--:|:--:|:--:|
|1|"ham"|"Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."|""|""|""|
|2|"ham"|"Ok lar... Joking wif u oni..."|""|""|""|
|3|"spam"|"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T\&C's apply 08452810075over18's"|""|""|""|
|4|"ham"|"U dun say so early hor... U c already then say..."|""|""|""|
|5|"ham"|"Nah I don't think he goes to usf, he lives around here though"|""|""|""|
|6|"spam"|"FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, �1.50 to rcv"|""|""|""|
|7|"ham"|"Even my brother is not like to speak with me. They treat me like aids patent."|""|""|""|
|8|"ham"|"As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"|""|""|""|



���ƂŁA�f�[�^�𕪊��������̂ŁA������ȗ������邽�߂ɁA�G�N�Z���t�@�C���̓��e�ł���ϐ�data��6��ڂɁA�J�e�S���J���^�ɕύX�������x�������i�[���܂��B


```matlab
data.event_type = categorical(data.v1);
```


���ɁA�f�[�^�Z�b�g����spam/ham�̊������~�O���t�ɂĕ\���܂��B


```matlab
f = figure;
pie(data.event_type,{'ham','spam'});
title("Class Distribution")
```

![figure_0.png](ClassifySpamMailUsingLSTM_Japanese_images/figure_0.png)

# �P���E���؁E�e�X�g�f�[�^�Z�b�g�ւ̕���


�܂��A�S�f�[�^��7�����P���f�[�^�Ƃ��Đ؂�o���܂��Bcvpartition�֐��ɁA�����قǂ�spam/ham���ł���data.event_type����͂��A�����̊�����0.3 (0.7) �Ƃ��܂��B���[�N�X�y�[�X�ɂ͌���܂��񂪁Atraining�Ƃ����ϐ��̂悤�Ȃ��̂ɁAcvp����͂���΁A�P���f�[�^�Ɋ���U����ׂ�����C���f�b�N�X��Ԃ��̂ŁA����𗘗p���āAdataTrain�𓾂܂��B


```matlab
cvp = cvpartition(data.event_type,'Holdout',0.3);
dataTrain = data(training(cvp),:);
dataHeldOut = data(test(cvp),:);
```


���l�ɁA�����قǂ̕����ł킯��ꂽ3���̂ق��̃f�[�^�����؃f�[�^�ƃe�X�g�f�[�^�ɕ������܂��B


```matlab
cvp = cvpartition(dataHeldOut.event_type,'HoldOut',0.5);
dataValidation = dataHeldOut(training(cvp),:);
dataTest = dataHeldOut(test(cvp),:);
```


��ŕ������f�[�^����A�w�K�ȂǂɎg�����߂̃e�L�X�g�f�[�^�⃉�x���������o���܂��B


```matlab
textDataTrain = dataTrain.v2;
textDataValidation = dataValidation.v2;
textDataTest = dataTest.v2;
YTrain = dataTrain.event_type;
YValidation = dataValidation.event_type;
YTest = dataTest.event_type;
```


wordcloud�֐��ŁA�P���f�[�^�Ɋ܂܂�Ă���P��₻�̕p�x���������܂��B�P��̑傫���́A���̕p�x�ɑΉ����Ă��܂��B


```matlab
figure
wordcloud(textDataTrain);
title("Training Data")
```

![figure_1.png](ClassifySpamMailUsingLSTM_Japanese_images/figure_1.png)

# �e�L�X�g�f�[�^�̑O����


���̃h�L�������g�̍Ō�ɕ⏕�֐��Ƃ��Ēu���Ă���`preprocessText`��p���āA�e�L�X�g�f�[�^�̑O�������s���Ă����܂��B




�Ⴆ�΁A�P���f�[�^�ł���4000���قǂ̃e�L�X�g�ɑ΂��āA�ȉ��̂R�̑�����s���܂��B




�P�D���ꂼ��̕��͂�����ɂ킯��B��j`an example of a short sentence => an + example + of + a + short + sentence`




2. �@���ꂼ��̕�������������������ɂ���@��jHello World => hello world




3.�@��Ǔ_��A�u �f �v������




�Ȃ��A����̉�͂ł͎��O�w�K�l�b�g���[�N���g�����߁Astop words�̏����͍s���܂���B


```matlab
documentsTrain = preprocessText(textDataTrain);
documentsValidation = preprocessText(textDataValidation);
documentsTest = preprocessText(textDataTest);
```


�������ď����������͂̂���5���Ƃ��ĕ\�����܂��B�啶����R���}���Ȃ����Ƃ��m�F�ł��܂��B


```matlab
documentsTrain(1:5)
```
```
ans = 
  5x1 tokenizedDocument:

    20 tokens: go until jurong point crazy available only in bugis n great world la e buffet cine there got amore wat
     6 tokens: ok lar joking wif u oni
    32 tokens: free entry in 2 a wkly comp to win fa cup final tkts 21st may 2005 text fa to 87121 to receive entry question std txt rate t cs apply 08452810075over18 s
    11 tokens: u dun say so early hor u c already then say
    13 tokens: nah i dont think he goes to usf he lives around here though

```
# �e�L�X�g�ւ̒ʂ��ԍ��̕t�^


����̗�ł́A�w�K�ς݂̃l�b�g���[�N���C���|�[�g���A�����ɓo�^����Ă���P��Əƍ������邱�Ƃł��ꂼ��̒P��Ɉ�ӂ̔w�ԍ���^���܂��B




����̗�ł́A���O�w�K�l�b�g���[�N���C���|�[�g���ifastText�j�A��������ƂɁA�P����x�N�g���ɕϊ����܂��B���̃x�N�g������������LSTM�l�b�g���[�N���w�K���܂��B




�Q�l�����FMikolov, Tomas, et al. "Advances in pre-training distributed word representations." *arXiv preprint arXiv:1712.09405* (2017).




�����ł͂܂��A��Ő����������O�w�K�l�b�g���[�N���C���|�[�g���܂��B




�����āA���̃l�b�g���[�N�ɓo�^����Ă��邻�ꂼ��̒P�ꂪ��ӂ̔w�ԍ������悤�ɂ��܂��B




`wordEncoding`�֐���p���邱�ƂŁA�P��Ɣԍ��̑Ή��֌W���쐬���܂��B


```matlab
emb = fastTextWordEmbedding;
enc = wordEncoding(tokenizedDocument(emb.Vocabulary,'TokenizeMethod','none'));
```


���ɁALSTM�̃l�b�g���[�N�ɓ��͂���f�[�^�i�������j�̏�����l���܂��B�������͂ł����Ă��A���f���[���̏ꍇ�͂��ׂēǂ܂��Ƃ��O���̂������̕��͂�ǂ߂΂킩��ꍇ�������Ɖ��肵�܂��B




�܂��A�\���ȏ���ۂ����܂܂ł���΁A�ł��邾���Z�����͂̂ق����w�K�����܂������₷���ł��B�����ŁA�P���f�[�^�̂��ꂼ��̕��͂����������ǂꂭ�炢�̒P�ꐔ�ō\������Ă��邩���m�F���܂��B




�܂��́A`doclength`�֐��ŌP���f�[�^�̂��ꂼ��̕��͂������̒P�� (token)�ō\������Ă��邩���v�Z���܂��B




�����āA�����̕��z��`histogram`�֐��Ŋm�F���邱�Ƃ��ł��܂��B


```matlab
documentLengths = doclength(documentsTrain);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")
```

![figure_2.png](ClassifySpamMailUsingLSTM_Japanese_images/figure_2.png)



��̕��z���݂�ƁA�قƂ�ǂ̕��͂��A75�P��ȉ��ł��邱�Ƃ��킩��܂��B�����Ŏ��̑���ł́A�P�ꐔ��75�𒴂���΁A�����I�ɂ����ŕ��͂��J�b�g����悤�ɂ��܂��B�ڂ����͈ȉ��Ő������܂��B




`doc2sequence`�֐���p���āA���ꂼ��̕��͂��A�P��̔w�ԍ��ŕ\���܂��B




�Ⴆ�΁A���͂��AI like baseball �ŁAI: 19, like: 78, baseball: 99 �̂悤�ɓo�^����Ă����ꍇ�́A




XTrain = [19 78 99]�̂悤�ȃx�N�g���ɕϊ�����܂��B  


```matlab
XTrain = doc2sequence(enc,documentsTrain,'Length',75);
XTrain(1:5)
```
| |1|
|:--:|:--:|
|1|1x75 double|
|2|1x75 double|
|3|1x75 double|
|4|1x75 double|
|5|1x75 double|



���؃f�[�^�Ɋւ��Ă����l�̑�����s���܂��B


```matlab
XValidation = doc2sequence(enc,documentsValidation,'Length',75);
XTest = doc2sequence(enc,documentsTest,'Length',75);
```
# LSTM�l�b�g���[�N�̍쐬


�w�K���s���ALSTM�l�b�g���[�N�̒�`���s���܂��B




`sequenceInputLayer`�œ��͑w���`���܂��BinputSize�͍���̏ꍇ�P�ł��B�Ⴆ�΁A�Z���T�[�̃f�[�^�i�C���A�����A���x�j�̎��n��f�[�^����͂Ƃ������ꍇ�A�Z���T�[�̐���inputSize�ɑ������܂��B����͂P�̃��[���̕��͂ɑ΂��āA�P�̃��x���i���f���[�����ۂ��j���Ή����Ă��܂��B




`wordEmbeddingLayer`�ł́A�����قǃC���|�[�g�������O�w�K�l�b�g���[�N�����ƂɁA���ꂼ��̒P�������x�N�g���ɕϊ����A�������̃f�[�^�ɕϊ����܂��B


```matlab
inputSize = 1;
words = emb.Vocabulary;
dimension = emb.Dimension;
numWords = numel(words);
numHiddenUnits = 180;
numClasses = numel(categories(YTrain));
layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(dimension,numWords,'Weights',word2vec(emb,words)')
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
```
```
layers = 
  ���̑w������ 6x1 �� Layer �z��:

     1   ''   �V�[�P���X����            1 �����̃V�[�P���X����
     2   ''   Word Embedding Layer   Word embedding layer with 300 dimensions and 999994 unique words
     3   ''   LSTM                   180 �B�ꃆ�j�b�g�̂��� LSTM
     4   ''   �S����                  2 �S�����w
     5   ''   �\�t�g�}�b�N�X            �\�t�g�}�b�N�X
     6   ''   ���ޏo��                 crossentropyex
```


�ȉ��Ɋw�K�̃I�v�V������ݒ肵�܂��B


```matlab
options = trainingOptions('adam', ...
    'MaxEpochs',6, ...    
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ExecutionEnvironment', 'auto', ...
    'Plots','training-progress', ...
    'Verbose',false);
```


`trainNetwork`�֐���p���āA�P�����s���܂��B


```matlab
net = trainNetwork(XTrain,YTrain,layers,options);
```

![figure_3.png](ClassifySpamMailUsingLSTM_Japanese_images/figure_3.png)

# �e�X�g�f�[�^�̗\��


��̌��،��ʂ��\���ł���΍Ō�ɏ�Ɠ��l�ɂ��ăe�X�g�f�[�^�̗\���₻�̕]�����s���Ă����܂��B




�O�̃Z�N�V�����ō쐬�����l�b�g���[�N`net`�ɑ΂��āA�e�X�g�f�[�^`XTest`��n���Ƃ���̗\�����x��`YPred`�𓾂邱�Ƃ��ł��܂��B


```matlab
YPred = classify(net,XTest);
```
| |1|
|:--:|:--:|
|1|1x75 double|
|2|1x75 double|
|3|1x75 double|
|4|1x75 double|
|5|1x75 double|



�S�̐��x�̌v�Z���s���܂��B�L���u==�v�́A�����\���Ɛ������x���������ł����1���A�����łȂ����0��Ԃ��܂��B���ꂪ�e�X�g�f�[�^�̐���������ł����܂��B���̂��߁A����1��0���̃x�N�g���̑S�v�f�̕��ς����ΐ��x���v�Z���邱�Ƃ��ł��܂��B


```matlab
accuracy = mean(YPred == YTest)
```
```
accuracy = 0.9916
```
# ���܂��F�����ō쐬�����e�L�X�g�̕���


�����ō쐬�������͂�����̕��ފ�ɂ�spam���ǂ������f�����邱�Ƃ��ł��܂��B�Ⴆ�Έȉ��̂悤��3�̕��͂�p�ӂ��܂��B


```matlab
NewMail = [ ...
    "please visit this webpage to get the special discount."
    "you can get cash after filling in the questionare."
    "please let me know when your paper is ready to submit."];
```


��قǂƓ��l�ɑO��������i�߂Ă����܂��B


```matlab
documentsNew = preprocessText(NewMail);
XNew = doc2sequence(enc,documentsNew,'Length',75);
[labelsNew,score] = classify(net,XNew);
```
```
ans = 3x2 �� string �z��    
"please��visit��this��webpage��t�c  "ham"        
"you��can��get��cash��after��fill�c  "ham"        
"please��let��me��know��when��you�c  "ham"        

```
```matlab
[reportsNew string(labelsNew)]
```
# �⏕�֐�
```matlab
function documents = preprocessText(textData)
% Tokenize the text.
documents = tokenizedDocument(textData);
% Convert to lowercase.
documents = lower(documents);
% Erase punctuation.
documents = erasePunctuation(documents);
end
```
