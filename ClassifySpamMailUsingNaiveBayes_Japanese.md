# �i�C�[�u�x�C�Y��p�������f���[���̌��o


���̗�ł́A�i�C�[�u�x�C�Y�ƌĂ΂���@��p���āA���̕��ʂ�����f���[�����ǂ����𔻕ʂ��܂��B




���̗�̂ق���LSTM(long short term memory)�ƌĂ΂���@��p�������̂�����܂��B




������̃i�C�[�u�x�C�Y��p�������@�Ɋւ��Ă͂�����̋L�����Q�l�ɂ����Ă��������܂����B




�@�B�w�K �� ���f���[�����ށi�i�C�[�u�x�C�Y���ފ�j ���@[https://qiita.com/fujin/items/50fe0e0227ef8457a473](https://qiita.com/fujin/items/50fe0e0227ef8457a473)




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
rng('default')
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

![figure_0.png](ClassifySpamMailUsingNaiveBayes_Japanese_images/figure_0.png)

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
wordcloud(textDataTrain)
```
```
ans = 
  WordCloudChart �̃v���p�e�B:

           WordData: [1x5747 string]
           SizeData: [1x5747 double]
    MaxDisplayWords: 100

  ���ׂẴv���p�e�B ��\��

```
```matlab
title("Training Data")
```

![figure_1.png](ClassifySpamMailUsingNaiveBayes_Japanese_images/figure_1.png)

# �e�L�X�g�f�[�^�̑O����


���̃h�L�������g�̍Ō�ɕ⏕�֐��Ƃ��Ēu���Ă���`preprocessText`��p���āA�e�L�X�g�f�[�^�̑O�������s���Ă����܂��B




�Ⴆ�΁A�P���f�[�^�ł���4000���قǂ̃e�L�X�g�ɑ΂��āA�ȉ��̂R�̑�����s���܂��B




�P�D���ꂼ��̕��͂�����ɂ킯��B��j`an example of a short sentence => an + example + of + a + short + sentence`




2. �@���ꂼ��̕�������������������ɂ���@��jHello World => hello world




3.�@��Ǔ_��A�u �f �v������


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

     6 tokens: ok lar joking wif u oni
    32 tokens: free entry in 2 a wkly comp to win fa cup final tkts 21st may 2005 text fa to 87121 to receive entry question std txt rate t cs apply 08452810075over18 s
    11 tokens: u dun say so early hor u c already then say
    13 tokens: nah i dont think he goes to usf he lives around here though
    33 tokens: freemsg hey there darling its been 3 weeks now and no word back id like some fun you up for it still tb ok xxx std chgs to send � 150 to rcv

```


����̂悤�ɕp�x���x�[�X�ɕ��ނ��s���ꍇ�AI��to�Ȃǂ̈�ʓI�ɍL���g����P��̕p�x�͂��܂蕪�ނɊ�^���Ȃ��ƍl�����܂��B�����ŁA�ȉ���sequence2freq�֐��ŏ�������ۂɁA�悭������P��̓J�E���g���Ȃ��悤�ɐ��䂵�܂�(stop word)�B




stopWords�Ƃ������O�ł��炩���ߍ폜���ׂ��P�ꂪ�p�ӂ���Ă��āAremoveWords�֐��ƕ��p���邱�Ƃł��ꂼ��̃e�L�X�g����stop words���폜���邱�Ƃ��ł��܂��B




([https://jp.mathworks.com/help/textanalytics/ref/stopwords.html](https://jp.mathworks.com/help/textanalytics/ref/stopwords.html))


```matlab
documentsTrain = removeWords(documentsTrain,stopWords);
documentsValidation = removeWords(documentsValidation,stopWords);
documentsTest = removeWords(documentsTest,stopWords);
documentsTrain(1:5)
```
```
ans = 
  5x1 tokenizedDocument:

     6 tokens: ok lar joking wif u oni
    26 tokens: free entry 2 wkly comp win fa cup final tkts 21st 2005 text fa 87121 receive entry question std txt rate t cs apply 08452810075over18 s
     9 tokens: u dun say early hor u c already say
     7 tokens: nah think goes usf lives around though
    21 tokens: freemsg hey darling 3 weeks word back id like fun up still tb ok xxx std chgs send � 150 rcv

```
# �e�L�X�g�̏o���p�x�̌v�Z


����̗�ł́A�P���f�[�^�Ŋϑ����ꂽ�S�P��𒲍����A���ꂼ��̒P��Ɉ�ӂ̔w�ԍ���^���܂��B




`wordEncoding`�֐��ɑ΂��āA�P���f�[�^����͂Ƃ��ė^���܂��B




�܂��A`'Order',"frequency"`�Ƃ���Γo�^����P��̏��Ԃ��A�P���f�[�^�Ŋϑ����ꂽ�p�x�̏��ԂɂȂ�܂��B


```matlab
enc = wordEncoding(documentsTrain,'Order',"frequency",'MaxNumWords',6000);
```


���ɁA`doc2sequence`�֐���p���āA���ꂼ��̕��͂��A�P��̔w�ԍ��ŕ\���܂��B




�Ⴆ�΁A���͂��AI like baseball �ŁAI: 19, like: 78, baseball: 99 �̂悤�ɓo�^����Ă����ꍇ�́A




XTrain = [19 78 99]�̂悤�ȃx�N�g���ɕϊ�����܂��B  


```matlab
XTrain = doc2sequence(enc,documentsTrain,'PaddingDirection','none');
```


�ϊ����`XTrain`�̈ꕔ��\�����܂��B�����̗���ŕ\������Ă��邱�Ƃ��킩��܂��B


```matlab
XTrain{3001:3003}
```
```
ans = 1x7    
         191         559         276          15         144        1686         798

ans = 1x18    
         291          15         267         421         186         258         591         363         734         103         199        1499          71         255          36         734         186          71

ans = 1x11    
           4        1573         126        1146        1802          86         336          47         448           4        2403

```


�����قǁAwordEncoding�֐����g�����ۂɁA�p�x�̏��ԂŒP���o�^����悤�ɐݒ肵�܂����B




ind2word�֐���p���āA�ϐ�enc�ɓo�^����Ă���P��̏��ԁi�C���f�b�N�X�j����A�ǂ̒P�ꂪ�o�^����Ă��邩���Q�Ƃ��邱�Ƃ��ł��܂��B���Ƃ��΁A�ȉ��̑���ōł������ϑ����ꂽ�P����20�����邱�Ƃ��ł��܂��B�Ȃ��Astop words�͍폜����Ă���̂ŁA�����͕\������܂���B


```matlab
idx = [1:20];
words = ind2word(enc,idx)
```
```
words = 1x20 �� string �z��    
"u"          "call"       "2"          "just"       "get"        "ur"         "�"          "gt"         "lt"         "up"         "4"          "ok"         "free"       "go"         "got"        "like"       ":)"         "good"       "come"       "know"       

```


���ɁA���̃h�L�������g�̍Ō�ɂ���⏕�֐�sequence2freq��p���āA���ꂼ��̕��͂ɁA�ǂ̒P�ꂪ����o�����������W�v���܂��B




�Ⴆ�΁A���͂��P��̔w�ԍ���p���āA[3 1 2 2 5 3]�Ƃ������͂ŕ\����Ă�����A���ꂼ��̒P��̕p�x�͈ȉ��̂悤�ɂȂ�܂��B




[1 2 2 0 1 0 0 0 ...]




�����ŁA�P���f�[�^�Ŋϑ����ꂽ�P��ɑ΂��Ē��ׂ���̂ŁA5�ȍ~�̒P��ɑ΂��Ă��p�x�̌v�Z���s���܂��i�p�x0���Ԃ���܂��j�B�P���f�[�^�Ŋϑ����ꂽ���P�ꐔ�ɑ΂��āA���ꂼ��̕��͂͏������̂ŁA������̕p�x�̃f�[�^�́A0�����ɑ����Ȃ�܂��B


```matlab
XTrainFreq=sequence2freq(XTrain,enc);
```


���l�Ɍ��؃f�[�^�E�e�X�g�f�[�^���������s���܂��B


```matlab
XValidation = doc2sequence(enc,documentsValidation,'PaddingDirection','none');
XValidationFreq=sequence2freq(XValidation,enc);
XTest = doc2sequence(enc,documentsTest,'PaddingDirection','none');
XTestFreq=sequence2freq(XTest,enc);
```


�ȏ�̑���ŁA�P���E���؁E�e�X�g�f�[�^�̂��ꂼ��̕��͂ɂ��āA�ǂ̒P�ꂪ�ǂꂭ�炢�̕p�x�ŏo�����邩���W�v���邱�Ƃ��ł��܂����B�p�x�Ƃ��������Ɩ��f���[�����ǂ����Ƃ������x�������ƂɁA�P���⌟�؂��s���Ă����܂��B


# �P���f�[�^�ɑ΂���A�i�C�[�u�x�C�Y�̎��s


`fitcnb`�֐��Ńi�C�[�u�x�C�Y��p�����P�����s�����Ƃ��ł��܂��B����̌P���f�[�^�͏�q�����悤��0�̑������̂ƂȂ��Ă��܂��B���̂��߁A���z�𑽍����z�����肵�܂��B`'DistributionNames','mn'`�Ƃ��Đ錾���邱�Ƃ��ł��܂��B�܂��A���O���z�͌P���f�[�^��spam/ham�̊������̗p���܂��B`'Prior','empirical'`�Ɛ錾����΂悢�ł��B


```matlab
Mdl = fitcnb(XTrainFreq,YTrain,'DistributionNames','mn','Prior','empirical');
```


predict�֐��ɁA��ō쐬�������f���ƁA���؃f�[�^����͂��邱�ƂŁA���؃f�[�^�̗\�����s�����Ƃ��ł��܂��B


```matlab
Ypred_Validation=predict(Mdl,XValidationFreq);
```


�����s����쐬���A�\�����e�̕��z���m�F���܂��B


```matlab
confusionchart(YValidation,Ypred_Validation)
```

![figure_2.png](ClassifySpamMailUsingNaiveBayes_Japanese_images/figure_2.png)

```
ans = 
  ConfusionMatrixChart �̃v���p�e�B:

    NormalizedValues: [2x2 double]
         ClassLabels: [2x1 categorical]

  ���ׂẴv���p�e�B ��\��

```
# �e�X�g�f�[�^�̗\��


��̌��،��ʂ��\���ł���΍Ō�ɏ�Ɠ��l�ɂ��ăe�X�g�f�[�^�̗\���₻�̕]�����s���Ă����܂��B


```matlab
[YPred_Test,Posterior,Cost]=predict(Mdl,XTestFreq);
confusionchart(YTest,YPred_Test)
```

![figure_3.png](ClassifySpamMailUsingNaiveBayes_Japanese_images/figure_3.png)

```
ans = 
  ConfusionMatrixChart �̃v���p�e�B:

    NormalizedValues: [2x2 double]
         ClassLabels: [2x1 categorical]

  ���ׂẴv���p�e�B ��\��

```
```matlab
Mdl.Prior
```
```
ans = 1x2    
    0.8659    0.1341

```
```matlab
accuracy = mean(YTest==YPred_Test)
```
```
accuracy = 0.9832
```


98�� �ȏ�̍������x�ŁA���f���[�����ǂ����𔻕ʂ��邱�Ƃ��ł��܂����B����̗�ł́A���ꂼ��̒P��̕p�x�����Ƃɕ��ނ��s���Ă��āA�P�ꓯ�m�̊֘A�⏇�ԂȂǂ͍l�����Ă��܂���B�����P�̗�ł���ALSTM��p�������ނł́A�P����x�N�g���ɕϊ����A���n��I�Ɉ����Ă��܂��B��낵����΂���������Q�Ƃ��������B


# ���܂��F�����ō쐬�����e�L�X�g�̕���


�����ō쐬�������͂�����̕��ފ�ɂ�spam���ǂ������f�����邱�Ƃ��ł��܂��B�Ⴆ�Έȉ��̂悤��3�̕��͂�p�ӂ��܂��B


```matlab
reportsNew = [ ...
    "please visit this webpage to get the special discount."
    "you can subscribe this online journal for free for one year"
    "please let me know when your paper is ready to submit."];
```


��قǂƓ��l�ɑO��������i�߂Ă����܂��B


```matlab
documentsNew = preprocessText(reportsNew);
XNew = doc2sequence(enc,documentsNew,'PaddingDirection','none');
XNewFreq=sequence2freq(XNew,enc);
```


`predict`�֐��ɓ��͂��܂��B


```matlab
[YPred_New,PosteriorNew,CostNew]=predict(Mdl,XNewFreq)
```
```
YPred_New = 3x1 �� categorical �z��    
spam         
ham          
ham          

PosteriorNew = 3x2    
    0.3643    0.6357
    0.6941    0.3059
    0.9998    0.0002

CostNew = 3x2    
    0.6357    0.3643
    0.3059    0.6941
    0.0002    0.9998

```


�ォ�珇�ɁAspam, spam, ham�Ɣ��f����Ă��邱�Ƃ��킩��܂��B


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

function freq=sequence2freq(sequence,enc)
numWords=enc.NumWords;
freq=zeros(numel(sequence),numWords-1);
edges = (1:numWords);
    for i=1:numel(sequence)
        freq(i,:)=histcounts(sequence{i},edges);    
    end
end
```
