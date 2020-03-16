# LSTM‚ð—p‚¢‚½–À˜fƒ[ƒ‹‚ÌŒŸo


‚±‚Ì—á‚Å‚ÍALSTM (Long Short Term Memory) ‚ÆŒÄ‚Î‚ê‚éŽè–@‚ð—p‚¢‚ÄA‚»‚Ì•¶–Ê‚©‚ç–À˜fƒ[ƒ‹‚©‚Ç‚¤‚©‚ð”»•Ê‚µ‚Ü‚·B




‚±‚Ì—á‚Ì‚Ù‚©‚ÉƒiƒC[ƒuƒxƒCƒY‚ÆŒÄ‚Î‚ê‚éŽè–@‚ð—p‚¢‚½‚à‚Ì‚à‚ ‚è‚Ü‚·B




‚Ü‚½A‚±‚¿‚ç‚ÌƒR[ƒh‚Ì‘‚«•û‚Í‰º‚ÌmatlabŒöŽ®ƒhƒLƒ…ƒƒ“ƒg‚ðŽQl‚É‚µ‚Ü‚µ‚½B




([https://jp.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html](https://jp.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html))


# ƒf[ƒ^‚ÌƒCƒ“ƒ|[ƒg


¡‰ñ—p‚¢‚éƒf[ƒ^‚ÍA‚±‚¿‚çi[https://www.kaggle.com/uciml/sms-spam-collection-dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) )‚É‚ ‚é




SMS Spam Collection Dataset Collection of SMS messages tagged as spam or legitimate@‚Æ‚¢‚¤ƒf[ƒ^ƒZƒbƒg‚Å‚·B




‡Œv‚ÅA5574Œ‚Ìƒ[ƒ‹/ƒXƒpƒ€ƒ[ƒ‹‚ª—pˆÓ‚³‚ê‚Ä‚¢‚Ü‚·Bã‚ÌURL‚©‚çƒf[ƒ^‚ðƒ_ƒEƒ“ƒ[ƒh‚·‚é‚ÆA




spam.csv‚Æ‚¢‚¤ƒtƒ@ƒCƒ‹‚ð“¾‚é‚±‚Æ‚ª‚Å‚«‚Ü‚·B‚»‚ê‚ð“Ç‚Ýž‚Ý‚Ü‚·B




ƒGƒNƒZƒ‹‚Éƒ‰ƒxƒ‹‚â‚»‚ê‚É‘Î‰ž‚·‚é•¶Í‚ª‹L˜^‚³‚ê‚Ä‚¢‚éê‡‚ÍAreadtableŠÖ”‚ðŽg‚¤‚Æ•Ö—˜‚Å‚·B




•Ï”–¼‚ðdata‚Æ‚µ‚ÄAƒGƒNƒZƒ‹ƒtƒ@ƒCƒ‹‚Ìî•ñ‚ð“Ç‚Ýž‚Ý‚Ü‚·B




headŠÖ”‚É‚Ä“Ç‚Ýž‚ñ‚¾ƒtƒ@ƒCƒ‹‚Ì“à—e‚Ìˆê•”‚ðŽèŒy‚ÉŠm”F‚Å‚«‚Ü‚·Bv1—ñ‚É–À˜fƒ[ƒ‹(spam)‚©‚»‚¤‚Å‚È‚¢‚©(ham)‚ª‘‚¢‚Ä‚¢‚Ü‚·B


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
|6|"spam"|"FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, å£1.50 to rcv"|""|""|""|
|7|"ham"|"Even my brother is not like to speak with me. They treat me like aids patent."|""|""|""|
|8|"ham"|"As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"|""|""|""|



‚ ‚Æ‚ÅAƒf[ƒ^‚ð•ªŠ„‚µ‚½‚¢‚Ì‚ÅA‚»‚ê‚ðŠÈ—ª‰»‚·‚é‚½‚ß‚ÉAƒGƒNƒZƒ‹ƒtƒ@ƒCƒ‹‚Ì“à—e‚Å‚ ‚é•Ï”data‚Ì6—ñ–Ú‚ÉAƒJƒeƒSƒŠƒJƒ‹Œ^‚É•ÏX‚µ‚½ƒ‰ƒxƒ‹î•ñ‚ðŠi”[‚µ‚Ü‚·B


```matlab
data.event_type = categorical(data.v1);
```


ŽŸ‚ÉAƒf[ƒ^ƒZƒbƒg’†‚Ìspam/ham‚ÌŠ„‡‚ð‰~ƒOƒ‰ƒt‚É‚Ä•\‚µ‚Ü‚·B


```matlab
f = figure;
pie(data.event_type,{'ham','spam'});
title("Class Distribution")
```

![figure_0.png](ClassifySpamMailUsingLSTM_Japanese_images/figure_0.png)

# ŒP—ûEŒŸØEƒeƒXƒgƒf[ƒ^ƒZƒbƒg‚Ö‚Ì•ªŠ„


‚Ü‚¸A‘Sƒf[ƒ^‚Ì7Š„‚ðŒP—ûƒf[ƒ^‚Æ‚µ‚ÄØ‚èo‚µ‚Ü‚·BcvpartitionŠÖ”‚ÉA‚³‚«‚Ù‚Ç‚Ìspam/hamî•ñ‚Å‚ ‚édata.event_type‚ð“ü—Í‚µA•ªŠ„‚ÌŠ„‡‚ð0.3 (0.7) ‚Æ‚µ‚Ü‚·Bƒ[ƒNƒXƒy[ƒX‚É‚ÍŒ»‚ê‚Ü‚¹‚ñ‚ªAtraining‚Æ‚¢‚¤•Ï”‚Ì‚æ‚¤‚È‚à‚Ì‚ÉAcvp‚ð“ü—Í‚·‚ê‚ÎAŒP—ûƒf[ƒ^‚ÉŠ„‚èU‚ç‚ê‚é‚×‚«‚·‚éƒCƒ“ƒfƒbƒNƒX‚ð•Ô‚·‚Ì‚ÅA‚»‚ê‚ð—˜—p‚µ‚ÄAdataTrain‚ð“¾‚Ü‚·B


```matlab
cvp = cvpartition(data.event_type,'Holdout',0.3);
dataTrain = data(training(cvp),:);
dataHeldOut = data(test(cvp),:);
```


“¯—l‚ÉA‚³‚«‚Ù‚Ç‚Ì•ªŠ„‚Å‚í‚¯‚ç‚ê‚½3Š„‚Ì‚Ù‚¤‚Ìƒf[ƒ^‚ðŒŸØƒf[ƒ^‚ÆƒeƒXƒgƒf[ƒ^‚É•ªŠ„‚µ‚Ü‚·B


```matlab
cvp = cvpartition(dataHeldOut.event_type,'HoldOut',0.5);
dataValidation = dataHeldOut(training(cvp),:);
dataTest = dataHeldOut(test(cvp),:);
```


ã‚Å•ª‚¯‚½ƒf[ƒ^‚©‚çAŠwK‚È‚Ç‚ÉŽg‚¤‚½‚ß‚ÌƒeƒLƒXƒgƒf[ƒ^‚âƒ‰ƒxƒ‹î•ñ‚ðŽæ‚èo‚µ‚Ü‚·B


```matlab
textDataTrain = dataTrain.v2;
textDataValidation = dataValidation.v2;
textDataTest = dataTest.v2;
YTrain = dataTrain.event_type;
YValidation = dataValidation.event_type;
YTest = dataTest.event_type;
```


wordcloudŠÖ”‚ÅAŒP—ûƒf[ƒ^‚ÉŠÜ‚Ü‚ê‚Ä‚¢‚é’PŒê‚â‚»‚Ì•p“x‚ð‰ÂŽ‹‰»‚µ‚Ü‚·B’PŒê‚Ì‘å‚«‚³‚ÍA‚»‚Ì•p“x‚É‘Î‰ž‚µ‚Ä‚¢‚Ü‚·B


```matlab
figure
wordcloud(textDataTrain);
title("Training Data")
```

![figure_1.png](ClassifySpamMailUsingLSTM_Japanese_images/figure_1.png)

# ƒeƒLƒXƒgƒf[ƒ^‚Ì‘Oˆ—


‚±‚ÌƒhƒLƒ…ƒƒ“ƒg‚ÌÅŒã‚É•â•ŠÖ”‚Æ‚µ‚Ä’u‚¢‚Ä‚¢‚é`preprocessText`‚ð—p‚¢‚ÄAƒeƒLƒXƒgƒf[ƒ^‚Ì‘Oˆ—‚ðs‚Á‚Ä‚¢‚«‚Ü‚·B




—á‚¦‚ÎAŒP—ûƒf[ƒ^‚Å‚ ‚é4000Œ‚Ù‚Ç‚ÌƒeƒLƒXƒg‚É‘Î‚µ‚ÄAˆÈ‰º‚Ì‚R‚Â‚Ì‘€ì‚ðs‚¢‚Ü‚·B




‚PD‚»‚ê‚¼‚ê‚Ì•¶Í‚ðŽš‹å‚É‚í‚¯‚éB—áj`an example of a short sentence => an + example + of + a + short + sentence`




2. @‚»‚ê‚¼‚ê‚Ì•ª‚¯‚½•¶Žš—ñ‚ð¬•¶Žš‚É‚·‚é@—ájHello World => hello world




3.@‹å“Ç“_‚âAu f v‚ðÁ‚·




‚È‚¨A¡‰ñ‚Ì‰ðÍ‚Å‚ÍŽ–‘OŠwKƒlƒbƒgƒ[ƒN‚ðŽg‚¤‚½‚ßAstop words‚Ìˆ—‚Ís‚¢‚Ü‚¹‚ñB


```matlab
documentsTrain = preprocessText(textDataTrain);
documentsValidation = preprocessText(textDataValidation);
documentsTest = preprocessText(textDataTest);
```


‚±‚¤‚µ‚Äˆ—‚µ‚½•¶Í‚Ì‚¤‚¿5‚Â‚ð—á‚Æ‚µ‚Ä•\Ž¦‚µ‚Ü‚·B‘å•¶Žš‚âƒRƒ“ƒ}‚ª‚È‚¢‚±‚Æ‚ªŠm”F‚Å‚«‚Ü‚·B


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
# ƒeƒLƒXƒg‚Ö‚Ì’Ê‚µ”Ô†‚Ì•t—^


¡‰ñ‚Ì—á‚Å‚ÍAŠwKÏ‚Ý‚Ìƒlƒbƒgƒ[ƒN‚ðƒCƒ“ƒ|[ƒg‚µA‚»‚±‚É“o˜^‚³‚ê‚Ä‚¢‚é’PŒê‚ÆÆ‡‚³‚¹‚é‚±‚Æ‚Å‚»‚ê‚¼‚ê‚Ì’PŒê‚ÉˆêˆÓ‚Ì”w”Ô†‚ð—^‚¦‚Ü‚·B




¡‰ñ‚Ì—á‚Å‚ÍAŽ–‘OŠwKƒlƒbƒgƒ[ƒN‚ðƒCƒ“ƒ|[ƒg‚µifastTextjA‚»‚ê‚ð‚à‚Æ‚ÉA’PŒê‚ðƒxƒNƒgƒ‹‚É•ÏŠ·‚µ‚Ü‚·B‚»‚ÌƒxƒNƒgƒ‹‚ð‚à‚¿‚¢‚ÄLSTMƒlƒbƒgƒ[ƒN‚ðŠwK‚µ‚Ü‚·B




ŽQl•¶Œ£FMikolov, Tomas, et al. "Advances in pre-training distributed word representations." *arXiv preprint arXiv:1712.09405* (2017).




‚±‚±‚Å‚Í‚Ü‚¸Aã‚Åà–¾‚µ‚½Ž–‘OŠwKƒlƒbƒgƒ[ƒN‚ðƒCƒ“ƒ|[ƒg‚µ‚Ü‚·B




‚»‚µ‚ÄA‚»‚Ìƒlƒbƒgƒ[ƒN‚É“o˜^‚³‚ê‚Ä‚¢‚é‚»‚ê‚¼‚ê‚Ì’PŒê‚ªˆêˆÓ‚Ì”w”Ô†‚ðŽ‚Â‚æ‚¤‚É‚µ‚Ü‚·B




`wordEncoding`ŠÖ”‚ð—p‚¢‚é‚±‚Æ‚ÅA’PŒê‚Æ”Ô†‚Ì‘Î‰žŠÖŒW‚ðì¬‚µ‚Ü‚·B


```matlab
emb = fastTextWordEmbedding;
enc = wordEncoding(tokenizedDocument(emb.Vocabulary,'TokenizeMethod','none'));
```


ŽŸ‚ÉALSTM‚Ìƒlƒbƒgƒ[ƒN‚É“ü—Í‚·‚éƒf[ƒ^i•¶Žš”j‚ÌãŒÀ‚ðl‚¦‚Ü‚·B’·‚¢•¶Í‚Å‚ ‚Á‚Ä‚àA–À˜fƒ[ƒ‹‚Ìê‡‚Í‚·‚×‚Ä“Ç‚Ü‚¸‚Æ‚à‘O”¼‚Ì‚¢‚­‚Â‚©‚Ì•¶Í‚ð“Ç‚ß‚Î‚í‚©‚éê‡‚ª‘½‚¢‚Æ‰¼’è‚µ‚Ü‚·B




‚Ü‚½A\•ª‚Èî•ñ‚ð•Û‚Á‚½‚Ü‚Ü‚Å‚ ‚ê‚ÎA‚Å‚«‚é‚¾‚¯’Z‚¢•¶Í‚Ì‚Ù‚¤‚ªŠwK‚ª‚¤‚Ü‚­‚¢‚«‚â‚·‚¢‚Å‚·B‚»‚±‚ÅAŒP—ûƒf[ƒ^‚Ì‚»‚ê‚¼‚ê‚Ì•¶Í‚ª‚¾‚¢‚½‚¢‚Ç‚ê‚­‚ç‚¢‚Ì’PŒê”‚Å\¬‚³‚ê‚Ä‚¢‚é‚©‚ðŠm”F‚µ‚Ü‚·B




‚Ü‚¸‚ÍA`doclength`ŠÖ”‚ÅŒP—ûƒf[ƒ^‚Ì‚»‚ê‚¼‚ê‚Ì•¶Í‚ª‚¢‚­‚Â‚Ì’PŒê (token)‚Å\¬‚³‚ê‚Ä‚¢‚é‚©‚ðŒvŽZ‚µ‚Ü‚·B




‚»‚µ‚ÄA‚»‚ê‚ç‚Ì•ª•z‚ð`histogram`ŠÖ”‚ÅŠm”F‚·‚é‚±‚Æ‚ª‚Å‚«‚Ü‚·B


```matlab
documentLengths = doclength(documentsTrain);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")
```

![figure_2.png](ClassifySpamMailUsingLSTM_Japanese_images/figure_2.png)



ã‚Ì•ª•z‚ð‚Ý‚é‚ÆA‚Ù‚Æ‚ñ‚Ç‚Ì•¶Í‚ªA75’PŒêˆÈ‰º‚Å‚ ‚é‚±‚Æ‚ª‚í‚©‚è‚Ü‚·B‚»‚±‚ÅŽŸ‚Ì‘€ì‚Å‚ÍA’PŒê”‚ª75‚ð’´‚¦‚ê‚ÎA‹­§“I‚É‚»‚±‚Å•¶Í‚ðƒJƒbƒg‚·‚é‚æ‚¤‚É‚µ‚Ü‚·BÚ‚µ‚­‚ÍˆÈ‰º‚Åà–¾‚µ‚Ü‚·B




`doc2sequence`ŠÖ”‚ð—p‚¢‚ÄA‚»‚ê‚¼‚ê‚Ì•¶Í‚ðA’PŒê‚Ì”w”Ô†‚Å•\‚µ‚Ü‚·B




—á‚¦‚ÎA•¶Í‚ªAI like baseball ‚ÅAI: 19, like: 78, baseball: 99 ‚Ì‚æ‚¤‚É“o˜^‚³‚ê‚Ä‚¢‚½ê‡‚ÍA




XTrain = [19 78 99]‚Ì‚æ‚¤‚ÈƒxƒNƒgƒ‹‚É•ÏŠ·‚³‚ê‚Ü‚·B  


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



ŒŸØƒf[ƒ^‚ÉŠÖ‚µ‚Ä‚à“¯—l‚Ì‘€ì‚ðs‚¢‚Ü‚·B


```matlab
XValidation = doc2sequence(enc,documentsValidation,'Length',75);
XTest = doc2sequence(enc,documentsTest,'Length',75);
```
# LSTMƒlƒbƒgƒ[ƒN‚Ìì¬


ŠwK‚ðs‚¤ALSTMƒlƒbƒgƒ[ƒN‚Ì’è‹`‚ðs‚¢‚Ü‚·B




`sequenceInputLayer`‚Å“ü—Í‘w‚ð’è‹`‚µ‚Ü‚·BinputSize‚Í¡‰ñ‚Ìê‡‚P‚Å‚·B—á‚¦‚ÎAƒZƒ“ƒT[‚Ìƒf[ƒ^i‹C‰·A•—‘¬AŽ¼“xj‚ÌŽžŒn—ñƒf[ƒ^‚ð“ü—Í‚Æ‚µ‚½‚¢ê‡AƒZƒ“ƒT[‚Ì”‚ªinputSize‚É‘Š“–‚µ‚Ü‚·B¡‰ñ‚Í‚P‚Â‚Ìƒ[ƒ‹‚Ì•¶Í‚É‘Î‚µ‚ÄA‚P‚Â‚Ìƒ‰ƒxƒ‹i–À˜fƒ[ƒ‹‚©”Û‚©j‚ª‘Î‰ž‚µ‚Ä‚¢‚Ü‚·B




`wordEmbeddingLayer`‚Å‚ÍA‚³‚«‚Ù‚ÇƒCƒ“ƒ|[ƒg‚µ‚½Ž–‘OŠwKƒlƒbƒgƒ[ƒN‚ð‚à‚Æ‚ÉA‚»‚ê‚¼‚ê‚Ì’PŒê‚ð‚ ‚éƒxƒNƒgƒ‹‚É•ÏŠ·‚µA‚ŽŽŸŒ³‚Ìƒf[ƒ^‚É•ÏŠ·‚µ‚Ü‚·B


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
  ŽŸ‚Ì‘w‚ð‚à‚Â 6x1 ‚Ì Layer ”z—ñ:

     1   ''   ƒV[ƒPƒ“ƒX“ü—Í            1 ŽŸŒ³‚ÌƒV[ƒPƒ“ƒX“ü—Í
     2   ''   Word Embedding Layer   Word embedding layer with 300 dimensions and 999994 unique words
     3   ''   LSTM                   180 ‰B‚êƒ†ƒjƒbƒg‚Ì‚ ‚é LSTM
     4   ''   ‘SŒ‹‡                  2 ‘SŒ‹‡‘w
     5   ''   ƒ\ƒtƒgƒ}ƒbƒNƒX            ƒ\ƒtƒgƒ}ƒbƒNƒX
     6   ''   •ª—Þo—Í                 crossentropyex
```


ˆÈ‰º‚ÉŠwK‚ÌƒIƒvƒVƒ‡ƒ“‚ðÝ’è‚µ‚Ü‚·B


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


`trainNetwork`ŠÖ”‚ð—p‚¢‚ÄAŒP—û‚ðs‚¢‚Ü‚·B


```matlab
net = trainNetwork(XTrain,YTrain,layers,options);
```

![figure_3.png](ClassifySpamMailUsingLSTM_Japanese_images/figure_3.png)

# ƒeƒXƒgƒf[ƒ^‚Ì—\‘ª


ã‚ÌŒŸØŒ‹‰Ê‚ª\•ª‚Å‚ ‚ê‚ÎÅŒã‚Éã‚Æ“¯—l‚É‚µ‚ÄƒeƒXƒgƒf[ƒ^‚Ì—\‘ª‚â‚»‚Ì•]‰¿‚ðs‚Á‚Ä‚¢‚«‚Ü‚·B




‘O‚ÌƒZƒNƒVƒ‡ƒ“‚Åì¬‚µ‚½ƒlƒbƒgƒ[ƒN`net`‚É‘Î‚µ‚ÄAƒeƒXƒgƒf[ƒ^`XTest`‚ð“n‚·‚Æ‚»‚ê‚Ì—\‘ªƒ‰ƒxƒ‹`YPred`‚ð“¾‚é‚±‚Æ‚ª‚Å‚«‚Ü‚·B


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



‘S‘Ì¸“x‚ÌŒvŽZ‚ðs‚¢‚Ü‚·B‹L†u==v‚ÍA‚à‚µ—\‘ª‚Æ³‰ðƒ‰ƒxƒ‹‚ª“¯‚¶‚Å‚ ‚ê‚Î1‚ðA‚»‚¤‚Å‚È‚¯‚ê‚Î0‚ð•Ô‚µ‚Ü‚·B‚»‚ê‚ªƒeƒXƒgƒf[ƒ^‚Ì”‚¾‚¯•À‚ñ‚Å‚¢‚«‚Ü‚·B‚»‚Ì‚½‚ßA‚»‚Ì1‚©0‚©‚ÌƒxƒNƒgƒ‹‚Ì‘S—v‘f‚Ì•½‹Ï‚ðŽæ‚ê‚Î¸“x‚ðŒvŽZ‚·‚é‚±‚Æ‚ª‚Å‚«‚Ü‚·B


```matlab
accuracy = mean(YPred == YTest)
```
```
accuracy = 0.9916
```
# ‚¨‚Ü‚¯FŽ©•ª‚Åì¬‚µ‚½ƒeƒLƒXƒg‚Ì•ª—Þ


Ž©•ª‚Åì¬‚µ‚½•¶Í‚ð¡‰ñ‚Ì•ª—ÞŠí‚É‚Äspam‚©‚Ç‚¤‚©”»’f‚³‚¹‚é‚±‚Æ‚ª‚Å‚«‚Ü‚·B—á‚¦‚ÎˆÈ‰º‚Ì‚æ‚¤‚É3‚Â‚Ì•¶Í‚ð—pˆÓ‚µ‚Ü‚·B


```matlab
NewMail = [ ...
    "please visit this webpage to get the special discount."
    "you can get cash after filling in the questionare."
    "please let me know when your paper is ready to submit."];
```


æ‚Ù‚Ç‚Æ“¯—l‚É‘Oˆ—“™‚ði‚ß‚Ä‚¢‚«‚Ü‚·B


```matlab
documentsNew = preprocessText(NewMail);
XNew = doc2sequence(enc,documentsNew,'Length',75);
[labelsNew,score] = classify(net,XNew);
```
```
ans = 3x2 ‚Ì string ”z—ñ    
"pleaseüüvisitüüthisüüwebpageüütc  "ham"        
"youüücanüügetüücashüüafterüüfillc  "ham"        
"pleaseüületüümeüüknowüüwhenüüyouc  "ham"        

```
```matlab
[reportsNew string(labelsNew)]
```
# •â•ŠÖ”
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
