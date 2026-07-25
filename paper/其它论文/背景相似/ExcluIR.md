TheThirty-NinthAAAIConferenceonArtificialIntelligence(AAAI-25)
|     |     | ExcluIR: |     | Exclusionary |     | Neural | Information |     |     | Retrieval |     |     |     |     |
| --- | --- | -------- | --- | ------------ | --- | ------ | ----------- | --- | --- | --------- | --- | --- | --- | --- |
WenhaoZhang1, MengqiZhang1*, ShiguangWu1, JiahuanPei2, ZhaochunRen3,
|     |     |     | MaartendeRijke4, |     |     | ZhuminChen1*, |     | PengjieRen1 |     |     |     |     |     |     |
| --- | --- | --- | ---------------- | --- | --- | ------------- | --- | ----------- | --- | --- | --- | --- | --- | --- |
1ShandongUniversity,Qingdao,China
2CentrumWiskunde&Informatica,Amsterdam,TheNetherlands
3LeidenUniversity,Leiden,TheNetherlands
4UniversityofAmsterdam,Amsterdam,TheNetherlands
{zhangwenhao,shiguang.wu}@mail.sdu.edu.cn,
{mengqi.zhang,chenzhumin,renpengjie}@sdu.edu.cn,
jiahuan.pei@cwi.nl,z.ren@liacs.leidenuniv.nl,m.derijke@uva.nl
|     |     |     | Abstract |     |     |     |     |     | Non-exclusionary query |     |     |     |     |     |
| --- | --- | --- | -------- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | --- | --- | --- |
What are the American sci-fi action movies
| Exclusion | is an important |     | and universal | linguistic | skill | that |     |     |     |     |     |     |     |     |
| --------- | --------------- | --- | ------------- | ---------- | ----- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
released in 2019?
humansusetoexpresswhattheydonotwant.Thereislittle
researchonexclusionaryretrieval,whereusersexpresswhat
Exclusionary query
theydonotwanttobepartoftheresultsproducedfortheir
queries.Weinvestigatethescenarioofexclusionaryretrieval What other sci-fi movies (besides Avengers:
indocumentretrievalforthefirsttime.WepresentExcluIR,  Endgame) were released in 2019?
asetofresourcesforexclusionaryretrieval,consistingofan
evaluationbenchmarkandatrainingsetforhelpingretrieval
modelstocomprehendexclusionaryqueries.Theevaluation
|     |     |     |     |     |     |     | Alita: Battle Angel |     |     |     |     | Avengers: Endgame |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | --- | --- | ----------------- | --- | --- |
benchmarkincludes3,452high-qualityexclusionaryqueries,
|     |     |     |     |     |     |     |     |     |     |     | Avengers:  |     | Endgame  | is  a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | -------- | ----- |
eachofwhichhasbeenmanuallyannotated.Thetrainingset Alita: Battle Angel is a 2019
|     |     |     |     |     |     |     | American  | cyberpunk  |     | action | 2019  | American  |     | superhero |
| --- | --- | --- | --- | --- | --- | --- | --------- | ---------- | --- | ------ | ----- | --------- | --- | --------- |
contains70,293exclusionaryqueries,eachpairedwithapos-
|     |     |     |     |     |     |     | film  | based  | on  | Yukito | film  | based  | on  the  | Marvel |
| --- | --- | --- | --- | --- | --- | --- | ----- | ------ | --- | ------ | ----- | ------ | -------- | ------ |
itivedocumentandanegativedocument.Weconductdetailed
|     |     |     |     |     |     |     | Kishiro's manga series Battle |     |     |     | Comics superhero team the |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | ------------------------- | --- | --- | --- |
experimentsandanalyses,obtainingthreemainobservations: Avengers...
Angel Alita...
(i)existingretrievalmodelswithdifferentarchitecturesstrug-
| gle to comprehend |     | exclusionary |     | queries effectively; |     | (ii) al- |           |              |     |         |                  |     |     |         |
| ----------------- | --- | ------------ | --- | -------------------- | --- | -------- | --------- | ------------ | --- | ------- | ---------------- | --- | --- | ------- |
|                   |     |              |     |                      |     |          | Figure 1: | A comparison |     | between | non-exclusionary |     |     | and ex- |
thoughintegratingourtrainingdatacanimprovetheperfor-
manceofretrievalmodelsonexclusionaryretrieval,therestill clusionaryqueries.Exclusionaryqueriesoftenspecifycon-
existsagapcomparedtohumanperformance;and(iii)gen- tenttobeexcluded(e.g.,“Avengers:Endgame”)toexpress
erativeretrievalmodelshaveanaturaladvantageinhandling theuser’srequirementsforomittingcertaininformation.In
exclusionaryqueries. thiscase,iftheretrievalsystemfailstocomprehendtheex-
|     |     |     |     |     |     |     | clusionary | nature | of a | query | (e.g., one | containing |     | the term |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ------ | ---- | ----- | ---------- | ---------- | --- | -------- |
“besides,”)itwillproduceretrievalresultsthatusersdonot
1 Introduction
desire.
| Selective attention |              | (Treisman   | 1964;  | LaBerge           | 1990; | Cherry     |     |     |     |     |     |     |     |     |
| ------------------- | ------------ | ----------- | ------ | ----------------- | ----- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020), defined      | as           | the ability | to     | focus on relevant |       | informa-   |     |     |     |     |     |     |     |     |
| tion while          | disregarding | irrelevant  |        | information,      | plays | a cru-     |     |     |     |     |     |     |     |     |
| cial role in        | shaping      | user’s      | search | behaviors.        | This  | principle, |     |     |     |     |     |     |     |     |
understandexclusionaryqueriescanpresentapotentiallyse-
originatingfromcognitivepsychology,notonlyshapeshu- riousproblem.Forexample,asshowninFigure1,imagine
| man perception | of  | the environment |     | but also | extends | its in- |     |     |     |     |     |     |     |     |
| -------------- | --- | --------------- | --- | -------- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
ausersearchingformoviesintheretrievalsystem.Heposes
| fluence to      | interactions | with             | information | retrieval |         | systems.  |          |            |          |        |              |          |              |           |
| --------------- | ------------ | ---------------- | ----------- | --------- | ------- | --------- | -------- | ---------- | -------- | ------ | ------------ | -------- | ------------ | --------- |
|                 |              |                  |             |           |         |           | a query  | like “What | other    | sci-fi | movies       | (besides |              | Avengers: |
| When searching  |              | for information, |             | users can | express | a de-     |          |            |          |        |              |          |              |           |
|                 |              |                  |             |           |         |           | Endgame) | were       | released | in     | 2019?”       | If the   | retrieval    | system    |
| sire to exclude | certain      | information.     |             | We refer  | to      | this phe- |          |            |          |        |              |          |              |           |
|                 |              |                  |             |           |         |           | cannot   | correctly  | address  | this   | exclusionary |          | requirement, | it        |
|                 | exclusionary |                  | retrieval,  |           |         |           |          |            |          |        |              |          |              |           |
nomenon as where users explicitly mayreturnresultscontainingcontentirrelevanttotheuser’s
indicatetheirpreferencetoexcludespecificinformation. interests(e.g.,themovie“Avengers:Endgame”),thusreduc-
Exclusionaryretrievalemphasizesacrucialneedforpre-
ingusersatisfaction.
| cision and | relevance | in information |     | retrieval. | It shows | how |          |     |              |     |           |         |     |            |
| ---------- | --------- | -------------- | --- | ---------- | -------- | --- | -------- | --- | ------------ | --- | --------- | ------- | --- | ---------- |
|            |           |                |     |            |          |     | Research | on  | exclusionary |     | retrieval | remains |     | relatively |
usersusetheirknowledgeandexpectationstofindinforma-
|     |     |     |     |     |     |     | rare. Early | studies | mainly | focus | on  | keyword-based |     | meth- |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------- | ------ | ----- | --- | ------------- | --- | ----- |
tionthatmeetstheirspecificneeds.Therefore,thefailureto
|     |     |     |     |     |     |     | ods (Nakkouzi |     | and Eastman |       | 1990;      | McQuire | and  | Eastman |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----------- | ----- | ---------- | ------- | ---- | ------- |
|     |     |     |     |     |     |     | 1998; Harvey  | et  | al. 2003).  | These | approaches |         | rely | on con- |
*Correspondingauthor.
Copyright©2025,AssociationfortheAdvancementofArtificial structingbooleanqueriesthatincludenegationterms,which
Intelligence(www.aaai.org).Allrightsreserved. is essentially a post-processing strategy. However, these
13295

| methods | have | limitations | due | to their | reliance | on struc- |     |     |     |     |     |     |     |
| ------- | ---- | ----------- | --- | -------- | -------- | --------- | --- | --- | --- | --- | --- | --- | --- |
turedqueries,makingthemunsuitableformorediverseand
|         |         |          |          |           |     |                |          |     | Document pairs |            |     | Synthetic query |     |
| ------- | ------- | -------- | -------- | --------- | --- | -------------- | -------- | --- | -------------- | ---------- | --- | --------------- | --- |
| complex | natural | language | queries. | Moreover, |     | post-retrieval | HotpotQA |     |                |            |     |                 |     |
|         |         |          |          |           |     |                |          |     |                | collection |     | generation      |     |
methods,suchasrule-basedfiltering,areimpracticalinreal-
| world applications, |     | because |     | they are | difficult | to optimize |          |          |     |     |     |     |     |
| ------------------- | --- | ------- | --- | -------- | --------- | ----------- | -------- | -------- | --- | --- | --- | --- | --- |
|                     |     |         |     |          |           |             | Positive | Negative |     |     |     |     |     |
end-to-end with other models and can introduce potential ExcluIR
|     |     |     |     |     |     |     | doc | doc |     |     |     | Query rewrite |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- |
side effects and instability to the final results. Although re- (synthetic)
Exclusionary query
centworkhasexploredtheimpactofnegationinmodernre-
trievalmodels(Rokach,Romano,andMaimon2008;Koop-
manetal.2010;Weller,Lawrie,andVanDurme2024),their Annotation Manual ExcluIR
|          |                  |     |     |              |           |        | guidelines |     |     | correction |     | benckmark |     |
| -------- | ---------------- | --- | --- | ------------ | --------- | ------ | ---------- | --- | --- | ---------- | --- | --------- | --- |
| focus is | on comprehending |     |     | the negation | semantics | within |            |     |     |            |     |           |     |
documentsratherthantheexclusionarynatureofqueries.
Figure2:OverviewofExcluIRdatasetconstructionprocess.
Atpresent,thereisnoevaluationdatasettoassesstheca-
pabilityofretrievalmodelsinexclusionaryretrieval.Toad-
dressthisgap,ourfirstcontributioninthispaperistheintro-
|         |                  |     |     |              |            |        | data instance | consisting |     | of two | interrelated |     | documents; (ii) |
| ------- | ---------------- | --- | --- | ------------ | ---------- | ------ | ------------- | ---------- | --- | ------ | ------------ | --- | --------------- |
| duction | of the resources |     | for | exclusionary | retrieval, | namely |               |            |     |        |              |     |                 |
ExcluIR. ExcluIR contains an evaluation benchmark to as- foreachdocumentpair,weemployChatGPTtogeneratean
sess the capability of retrieval models in exclusionary re- exclusionaryquery.(iii)toenhancethediversityofthesyn-
|     |     |     |     |     |     |     | thetic queries, | we  | further | use | ChatGPT | to  | rephrase them; |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ------- | --- | ------- | --- | -------------- |
trieval,whilealsoprovidingatrainingdatasetthatincludes
|              |          |     |             |     |       |               | and (iv) | finally, | to ensure | a high | quality | of  | the dataset, we |
| ------------ | -------- | --- | ----------- | --- | ----- | ------------- | -------- | -------- | --------- | ------ | ------- | --- | --------------- |
| exclusionary | queries. |     | The dataset | is  | built | based on Hot- |          |          |           |        |         |     |                 |
establishannotationguidelinesandhireworkersformanual
potQA(Yangetal.2018).WefirstuseChatGPT1togenerate
| anexclusionaryqueryfortwogivenrelevantdocuments,re- |           |     |          |          |     |              | correction. |     |     |     |     |     |     |
| --------------------------------------------------- | --------- | --- | -------- | -------- | --- | ------------ | ----------- | --- | --- | --- | --- | --- | --- |
| quiring                                             | that only | one | document | contains | the | answer while |             |     |     |     |     |     |     |
2.1 Collectingdocumentspairs
| explicitly | rejecting | information |     | from | the other | document. |     |     |     |     |     |     |     |
| ---------- | --------- | ----------- | --- | ---- | --------- | --------- | --- | --- | --- | --- | --- | --- | --- |
Subsequently,weemploy17workerstocheckeachdatain- Webegintheconstructionprocessbycollectingdocuments
stanceinthebenchmarktoensuredataquality.Thetraining fromtheHotpotQA(Yangetal.2018)dataset,whichisde-
setcomprises70,293exclusionaryqueries,whilethebench- signed for multi-hop reasoning in question-answering task.
markincludes3,452human-annotatedexclusionaryqueries. Eachdatainstanceincludestwosupportingdocumentsthat
arerelated.Themodelneedstocomprehendtheassociation
Thisdatasetcanevaluatewhetherretrievalmodelscancor-
betweenthemandextractinformationfromthemtoanswer
| rectly retrieve | documents |     | when | dealing | with | exclusionary |     |     |     |     |     |     |     |
| --------------- | --------- | --- | ---- | ------- | ---- | ------------ | --- | --- | --- | --- | --- | --- | --- |
queries,providinganewperspectiveforevaluatingretrieval the question. We extract two related documents from each
models. data instance to form our document pairs. In total, we col-
Our second contribution is to analyze the performance lected74,293documentpairs.Aftermergingandremoving
|     |     |     |     |     |     |     | duplicates, | we  | obtained | a document |     | collection | containing |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | -------- | ---------- | --- | ---------- | ---------- |
ofexistingretrievalmethodswithdifferentarchitectureson
90,406documents.
exclusionaryretrieval,includingsparseretrieval(Robertson
| and Zaragoza | 2009; | Nogueira, |     | Lin, | and Epistemic | 2019), |     |     |     |     |     |     |     |
| ------------ | ----- | --------- | --- | ---- | ------------- | ------ | --- | --- | --- | --- | --- | --- | --- |
2.2 Generatingexclusionaryqueries
denseretrieval(Karpukhinetal.2020;Nietal.2022a),and
generativeretrievalmethods(Bevilacquaetal.2022;Wang Toefficientlyconstructourdataset,wedesignapromptcare-
et al. 2022a). We conduct extensive experiments and ar- fully to guide ChatGPT in generating exclusionary queries
rive at three main observations: (i) Existing retrieval mod- for each pair of documents. To ensure that the generated
els cannot fully understand the real intent of exclusion- queriescoverbothpositiveandnegativedocuments,wede-
aryqueries;(ii)Generativeretrievalmodelspossessunique sign a two-step construction strategy. Specifically, we first
advantages in exclusionary retrieval, while late interaction instruct ChatGPT to generate a query relevant to both doc-
modelslikeColBERThaveobviouslimitationsinhandling
|     |     |     |     |     |     |     | uments, | and then | guide | ChatGPT | to  | revise | this query by |
| --- | --- | --- | --- | --- | --- | --- | ------- | -------- | ----- | ------- | --- | ------ | ------------- |
suchqueries;(iii)Fine-tuningtheretrievalmodelswiththe addingaconstrainttoincludethesemanticsofrefusaltoin-
trainingsetofExcluIRcanimprovetheperformanceonex- formationfromthenegativedocument.
| clusionary | retrieval, | but | the | results are | still | far from sat- |     |     |     |     |     |     |     |
| ---------- | ---------- | --- | --- | ----------- | ----- | ------------- | --- | --- | --- | --- | --- | --- | --- |
isfactory. We provide in-depth analyses of these observa- 2.3 Rewritingsyntheticqueries
tions.Theseconclusionscontributevaluableinsightsforfu-
|     |     |     |     |     |     |     | Although | the prompt | has | been | carefully | adjusted, | the gen- |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | --- | ---- | --------- | --------- | -------- |
ture research on addressing the challenges of exclusionary erated queries often express the exclusionary phrases in a
retrieval.Wesharethebenchmarkandevaluationscriptson limitedmanner,suchas“excludinganyinformationabout,”
https://github.com/zwh-sdu/ExcluIR. “exceptforanyinformation,”and“withoutreferencingany
informationabout.”Theseexpressionslacknaturalnessand
2 DatasetConstruction
|     |     |     |     |     |     |     | deviate from | real-world |     | queries. | To  | increase | the diversity |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ---------- | --- | -------- | --- | -------- | ------------- |
andnaturalnessofthequeries,wefurtherinstructChatGPT
| As depicted | in  | Figure | 2, the | construction | of  | the ExcluIR |     |     |     |     |     |     |     |
| ----------- | --- | ------ | ------ | ------------ | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
torephrasethem.Then,wepartitiontheExcluIRdatasetob-
datasetinvolvesthefollowingsteps:(i)wefirstextractdoc-
|     |     |     |     |     |     |     | tained in | this step | into | training | and | test sets. | The test set |
| --- | --- | --- | --- | --- | --- | --- | --------- | --------- | ---- | -------- | --- | ---------- | ------------ |
umentpairsfromHotpotQA(Yangetal.2018),whereeach
|     |     |     |     |     |     |     | is further | manually | corrected |     | to construct |     | the benchmark, |
| --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | --------- | --- | ------------ | --- | -------------- |
1https://platform.openai.com/docs/models/gpt-3-5 whichwewilldescribenext.
13296

|     |     |     |     |     |     |     | training | set and | benchmark | are | 22.37 | and | 21.64, | respec- |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------- | --------- | --- | ----- | --- | ------ | ------- |
tively.Tofurtherinvestigatethediversityofdata,wevisual-
izethedistributionofthelengthsofexclusionaryqueriesin
|     |     |     |     |     |     |     | Figure 3.    | We show    | that    | the lengths | of     | exclusionary  |     | queries |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ---------- | ------- | ----------- | ------ | ------------- | --- | ------- |
|     |     |     |     |     |     |     | are diverse, | reflecting | varying |             | levels | of complexity | and | de-     |
tails.
3 ExperimentalSetup
|     |     |     |     |     |     |     | Methods | for comparison. |     | To  | evaluate | the performance |     | of  |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------------- | --- | --- | -------- | --------------- | --- | --- |
variousretrievalmodelsonexclusionaryretrieval,weselect
Figure3:Distributionofthelengthsofexclusionaryqueries three types of retrieval models with different architectures:
inExcluIR. sparseretrieval,denseretrieval,andgenerativeretrieval.
|                            |     |     |     |     |     |     | Sparse                       | retrieval | methods    | calculate |     | the relevance |      | score  |
| -------------------------- | --- | --- | --- | --- | --- | --- | ---------------------------- | --------- | ---------- | --------- | --- | ------------- | ---- | ------ |
|                            |     |     |     |     |     |     | of documents                 |           | using term | matching  |     | metrics       | such | as TF- |
| 2.4 Manuallycorrectingdata |     |     |     |     |     |     | IDF(RobertsonandWalker1997). |           |            |           |     |               |      |        |
TobuildareliableExcluIRbenchmark,wehire17workers • BM25 (Robertson and Zaragoza 2009) is a classical
formanualdatacorrection.Wefirstsample4,000instances probabilisticretrievalmethodbasedonthenormalization
ofthefrequencyofthetermandthelengthofthedocu-
fromthe74,293exclusionaryqueriesobtainedintheprevi-
ment.
ousstep.Eachinstancecontainstwodocumentsalongwith
asyntheticquerygeneratedbyChatGPT.Weaskworkersto • DocT5Query (Nogueira, Lin, and Epistemic 2019) ex-
checkthesyntheticexclusionaryquerytoensureitsnatural- pands documents by generating pseudo queries using
nessandcorrectnessandtheyareencouragedtoexpressthe
|     |     |     |     |     |     |     | a fine-tuned |     | T5 model | before | building |     | the BM25 | in- |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | -------- | ------ | -------- | --- | -------- | --- |
exclusionarynatureofqueriesusingdiverseexpressions.To dex(Raffeletal.2020).
facilitatethecorrectionprocess,weconstructanonlinecor-
Denseretrievalusespre-trainedlanguagemodels(PLMs)
rectionsystem.Inthesystem,wedefinethreeoperationsfor
|     |     |     |     |     |     |     | as the backbones |     | to represent |     | queries | and | documents | as  |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ------------ | --- | ------- | --- | --------- | --- |
workerstocorrecteachdatainstance:
densevectorsforcomputingrelevancescores.
1. CriteriaMet.Ifthesyntheticqueryalreadymeetsthecri-
|     |     |     |     |     |     |     | • DPR | (Karpukhin | et  | al. 2020) | is a | dense | retrieval | model |
| --- | --- | --- | --- | --- | --- | --- | ----- | ---------- | --- | --------- | ---- | ----- | --------- | ----- |
teria,nofurthermodificationsarenecessary. based on dual-encoder architecture, which uses the rep-
| 2. Query                                          | Modification. |     | If the | synthetic | query | fails to meet |             |     |              |            |      |              |         |        |
| ------------------------------------------------- | ------------- | --- | ------ | --------- | ----- | ------------- | ----------- | --- | ------------ | ---------- | ---- | ------------ | ------- | ------ |
|                                                   |               |     |        |           |       |               | resentation |     | of the [CLS] | token      | of   | BERT         | (Devlin | et al. |
| thecriteria,modifyorrewritethequerytoalignwiththe |               |     |        |           |       |               | 2019).      |     |              |            |      |              |         |        |
| requirements.                                     |               |     |        |           |       |               | Sentence-T5 |     |              |            |      |              |         |        |
|                                                   |               |     |        |           |       |               | •           |     | (Ni et       | al. 2022a) | uses | a fine-tuned |         | T5 en- |
3. DiscardData.Ifitisdifficulttowriteaquerythatmeets codermodeltoencodequeriesanddocumentsintodense
| the criteria |     | based on | these | two documents, |     | the workers | vectors. |     |     |     |     |     |     |     |
| ------------ | --- | -------- | ----- | -------------- | --- | ----------- | -------- | --- | --- | --- | --- | --- | --- | --- |
canchoosetodiscardthedata.
|     |     |     |     |     |     |     | • GTR       | (Ni | et al. 2022b) | has  | the        | same | architecture | as      |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------------- | ---- | ---------- | ---- | ------------ | ------- |
|     |     |     |     |     |     |     | Sentence-T5 |     | and has       | been | pretrained | on   | two          | billion |
2.5 Qualityassurance question-answerpairscollectedfromtheWeb.
Wetakeseveralmeasurestoensuredataquality:(i)wepro- • ColBERT (Khattab and Zaharia 2020) is a late inter-
|     |     |     |     |     |     |     | action | model | that learns | embeddings |     | for | each token | in  |
| --- | --- | --- | --- | --- | --- | --- | ------ | ----- | ----------- | ---------- | --- | --- | ---------- | --- |
videdetaileddocumentationguidelines,includingtaskdef-
queriesanddocuments,andthenusesaMaxSimopera-
| inition, | correction | process, | and | specific | criteria | for exclu- |     |     |     |     |     |     |     |     |
| -------- | ---------- | -------- | --- | -------- | -------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
tortocalculatetherelevancescore.
sionaryqueries;(ii)wepresentmultipleexamplesofexclu-
sionaryqueriestohelpworkersunderstandthetaskandre- Generativeretrievalisanend-to-endretrievalparadigm.
| quirements; | (iii) | we record | a video | to  | demonstrate | the en- |         |     |        |           |           |          |     |      |
| ----------- | ----- | --------- | ------- | --- | ----------- | ------- | ------- | --- | ------ | --------- | --------- | -------- | --- | ---- |
|             |       |           |         |     |             |         | • GENRE | (De | Cao et | al. 2020) | retrieves | entities | by  | gen- |
tirecorrectionprocessandemphasizethekeyconsiderations
|                |         |            |         |          |             |             | erating    | their | names       | through   | a seq-to-seq |             | model,     | it can |
| -------------- | ------- | ---------- | ------- | -------- | ----------- | ----------- | ---------- | ----- | ----------- | --------- | ------------ | ----------- | ---------- | ------ |
| that need      | special | attention; | (iv)    | we adopt | a real-time | feed-       |            |       |             |           |              |             |            |        |
|                |         |            |         |          |             |             | be applied |       | to document | retrieval |              | by directly | generating |        |
| back mechanism |         | to allow   | workers | to       | share the   | issues they |            |       |             |           |              |             |            |        |
documenttitles.TheoriginalGENREistrainedbasedon
| encounter | during | the correction |     | process; | we discuss | these |     |     |     |     |     |     |     |     |
| --------- | ------ | -------------- | --- | -------- | ---------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
BARTasthebackbone,andwereproduceitusingT5.
| issues and   | provide | solutions | accordingly; |         | and    | (v) we ran- |        |             |     |              |           |     |           |     |
| ------------ | ------- | --------- | ------------ | ------- | ------ | ----------- | ------ | ----------- | --- | ------------ | --------- | --- | --------- | --- |
|              |         |           |              |         |        |             | • SEAL | (Bevilacqua |     | et al. 2022) | retrieves |     | documents | by  |
| domly sample |         | 10% of    | the data     | of each | worker | for quality |        |             |     |              |           |     |           |     |
generatingn-gramswithinthem.
| inspection. | If there | are | errors | in the sampled | data, | we will |     |     |     |     |     |     |     |     |
| ----------- | -------- | --- | ------ | -------------- | ----- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
asktheworkertocorrectthedataagain. • NCI(Wangetal.2022a)proposesaprefix-awareweight-
adaptivedecoderarchitecture,leveragingsemanticdocu-
2.6 Datasetstatistics mentidentifiersandvariousdataaugmentationstrategies
likequerygeneration.
Followingthedatasetconstructionprocessdescribedabove,
we obtained 3,452 human-annotated entries for the bench- Evaluationmetrics.Fortheoriginaltestqueries,wereport
mark and 70,293 exclusionary queries for the training set. thecommonlyusedmetrics:RecallatrankN (R@N,N =
The average word counts for exclusionary queries in the 1,5,10)andMeanReciprocalRankatrankN (MRR@N,
13297

|     |      |       |     |     |     | HotpotQA |     |     |     |     | ExcluIR |      |     |     |
| --- | ---- | ----- | --- | --- | --- | -------- | --- | --- | --- | --- | ------- | ---- | --- | --- |
|     | Type | Model |     |     |     |          |     |     |     |     |         |      |     |     |
|     |      |       |     | R@2 | R@5 | R@10     | MRR | R@1 |     | MRR | ∆R@1    | ∆MRR | RR  |     |
Sparse BM25 67.16 76.65 80.98 92.47 49.68 65.17 7.82 4.66 53.48
Retrieval DocT5Query 69.19 77.88 81.65 94.10 50.98 67.50 7.85 3.81 53.85
|     |     | DPR |     | 55.53 | 67.44 | 73.49 | 81.73 | 49.63 | 65.79 |     | 7.34 | 5.01 | 54.02 |     |
| --- | --- | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ---- | ---- | ----- | --- |
Dense Sentence-T5 57.63 68.45 74.29 82.48 51.04 66.27 10.11 7.01 55.41
Retrieval GTR 61.82 73.57 79.42 85.50 54.87 70.88 14.40 8.79 57.42
|     |     | ColBERT |     | 73.58 | 83.73 | 87.95 | 94.44 | 54.00 | 71.24 |     | 10.72 | 6.42 | 55.57 |     |
| --- | --- | ------- | --- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ---- | ----- | --- |
|     |     | GENRE   |     | 48.87 | 51.67 | 53.24 | 75.25 | 48.03 | 63.22 |     | 4.35  | 0.13 | 52.10 |     |
Generative
|     |     | SEAL |     | 60.78 | 72.26 | 78.20 | 85.76 | 51.33 | 67.88 |     | 11.64 | 7.71 | 55.52 |     |
| --- | --- | ---- | --- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ---- | ----- | --- |
Retrieval
|     |     | NCI |     | 47.60 | 58.14 | 64.37 | 74.59 | 37.22 | 51.37 |     | 1.97 | 2.29 | 50.93 |     |
| --- | --- | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ---- | ---- | ----- | --- |
Table1:PerformanceofmodelstrainedonHotpotQAandtestedonHotpotQAandExcluIR.FortheevaluationonHotpotQA,
wereportRecall@2ratherthanRecall@1,sinceeachqueryinHotpotQAhastwosupportingdocuments.
|     |      |        |     |     |     | NQ320k |     |     |     |     | ExcluIR |      |     |     |
| --- | ---- | ------ | --- | --- | --- | ------ | --- | --- | --- | --- | ------- | ---- | --- | --- |
|     | Type | Method |     |     |     |        |     |     |     |     |         |      |     |     |
|     |      |        |     | R@1 | R@5 | R@10   | MRR | R@1 |     | MRR | ∆R@1    | ∆MRR | RR  |     |
Sparse BM25 37.96 61.24 68.86 47.86 49.68 65.17 7.82 4.66 53.48
Retrieval DocT5Query 42.63 66.18 73.38 52.69 50.98 67.50 7.85 3.81 53.85
|     |     | DPR |     | 54.81 | 79.50 | 85.52 | 65.39 | 48.55 | 60.50 |     | 16.45 | 13.49 | 58.76 |     |
| --- | --- | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ----- | ----- | --- |
Dense Sentence-T5 59.63 82.78 87.42 69.57 57.76 66.34 32.90 27.96 67.83
Retrieval GTR 62.35 84.67 89.17 71.90 59.79 69.00 34.85 28.12 68.31
ColBERT 60.08 84.19 89.41 70.50 57.01 70.88 20.02 15.26 59.97
|     |     | GENRE |     | 56.25 | 71.21 | 74.00 | 62.80 | 31.63 | 37.63 |     | 11.44 | 10.15 | 58.65 |     |
| --- | --- | ----- | --- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ----- | ----- | --- |
Generative
|     |     | SEAL |     | 55.24 | 75.13 | 80.97 | 63.86 | 43.54 | 55.17 |     | 16.11 | 15.27 | 60.02 |     |
| --- | --- | ---- | --- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ----- | ----- | --- |
Retrieval
|     |     | NCI |     | 60.41 | 76.10 | 80.19 | 67.18 | 31.46 | 38.95 |     | 15.87 | 16.81 | 56.84 |     |
| --- | --- | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ----- | ----- | --- |
Table2:PerformanceofmodelstrainedonNQ320kandtestedonNQ320kandExcluIR.
| N = 10). | Recall             | measures | the proportion |            | of relevant | doc-   | (Section4.4). |     |     |     |     |     |     |     |
| -------- | ------------------ | -------- | -------------- | ---------- | ----------- | ------ | ------------- | --- | --- | --- | --- | --- | --- | --- |
| uments   | that are retrieved |          | in the top     | N results. | MRR         | is the |               |     |     |     |     |     |     |     |
meanofthereciprocaloftherankofthefirstrelevantdocu- 4.1 Howwelldoexistingmethodsperformon
ment.
ExcluIR?
InExcluIR,eachexclusionaryqueryqhasapositivedoc-
|     |     |     |     |     |     |     | To  | evaluate | the | performance | of  | various | retrieval | models |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | ----------- | --- | ------- | --------- | ------ |
umentd+andanegativedocumentd−.Thus,thedifference
trainedonexistingdatasetsinExcluIR,weconductourex-
betweentherankofd+andtherankofd−canreflectthere-
|                 |            |     |                  |     |     |            | periments |     | on two | well-known |     | standard | retrieval | datasets: |
| --------------- | ---------- | --- | ---------------- | --- | --- | ---------- | --------- | --- | ------ | ---------- | --- | -------- | --------- | --------- |
| trieval model’s | capability |     | of comprehending |     | the | exclusion- |           |     |        |            |     |          |           |           |
aryquery.Sowereport∆R@N and∆MRR@N,whichcan NaturalQuestions(NQ)(Kwiatkowskietal.2019)andHot-
|     |     |     |     |     |     |     | potQA | (Yang | et  | al. 2018). | NQ  | is a large-scale |     | dataset for |
| --- | --- | --- | --- | --- | --- | --- | ----- | ----- | --- | ---------- | --- | ---------------- | --- | ----------- |
beformulatedas:
documentretrievalandquestionanswering.Theversionwe
∆R@N=R@N(d+)−R@N(d−),
|     |     |     |     |     |     |     | use | is NQ320k, |     | which | consists | of 320k | query-document |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ----- | -------- | ------- | -------------- | --- |
(1)
∆MRR@N=MRR@N(d+)−MRR@N(d−). pairs. HotpotQA is a question-answering dataset that fo-
|              |            |       |              |        |          |          | cuses | on  | multi-hop | reasoning. |        | We split | the original | Hot-      |
| ------------ | ---------- | ----- | ------------ | ------ | -------- | -------- | ----- | --- | --------- | ---------- | ------ | -------- | ------------ | --------- |
| In addition, | we report  | Right | Rank         | (RR),  | which is | the pro- |       |     |           |            |        |          |              |           |
|              |            |       |              |        |          |          | potQA | in  | the same  | way        | as our | ExcluIR  | dataset,     | resulting |
| portion      | of results | where | d+ is ranked | higher | than     | d−. The  |       |     |           |            |        |          |              |           |
ina70ktrainingsetanda3.5ktestset.
expectedvalueofRRis50%withrandomranking.
|     |     |     |     |     |     |     |     | The main | performance |     | of retrieval | methods |     | on the Ex- |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | --- | ------------ | ------- | --- | ---------- |
cluIRbenchmarkandothertestdataarepresentedinTable1
4 ResultsandAnalyses and2.Wehavethefollowingobservationsfromtheresults.
In this section, we present four groups of experimental re- First, although these methods achieve good performance
sultsandanalysestostudy:(i)theperformanceoftheexist- onthestandardtestdataincludingHotpotQAandNQ320k,
ingretrievalmodelsonExcluIR(Section4.1),(ii)thestrat- their performance on the ExcluIR benchmark is unsat-
egy to improve the performance on ExcluIR, including in- isfactory. Nearly all models score less than 10% higher
corporating our dataset into the training data (Section 4.2), than random ranking on the RR metric. Despite the fact
andscaling upthemodel size(Section4.3), (iii)the expla- that the Sentence-T5 and GTR models trained on NQ320k
nationforthesuperiorityofgenerativeretrievalinExcluIR achieve the highest ∆R@1/∆MRR/RR scores, they are far
13298

|       |     |              |     |     |       | NQ320k |       |       |       |       | ExcluIR |      |       |       |     |
| ----- | --- | ------------ | --- | --- | ----- | ------ | ----- | ----- | ----- | ----- | ------- | ---- | ----- | ----- | --- |
| Model |     | TrainingData |     |     |       |        |       |       |       |       |         |      |       |       |     |
|       |     |              |     |     | R@1   | R@5    | R@10  | MRR   | R@1   | MRR   | ∆R@1    | ∆MRR |       | RR    |     |
|       |     | NQ320k       |     |     | 54.81 | 79.50  | 85.52 | 65.39 | 48.55 | 60.50 | 16.45   |      | 13.49 | 58.76 |     |
DPR
N.w/ExcluIR 55.08 79.31 85.49 65.58 55.04† 67.89† 21.52† 16.38† 61.00†
|     |     | NQ320k |     |     | 59.63 | 82.78 | 87.42 | 69.57 | 57.76 | 66.34 | 32.90 |     | 27.96 | 67.83 |     |
| --- | --- | ------ | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ----- | --- |
Sentence-T5
N.w/ExcluIR 59.80 81.58 87.13 69.36 63.09† 74.57† 34.47† 26.19 68.00†
|     |     | NQ320k |     |     | 62.35 | 84.67 | 89.17 | 71.90 | 59.79 | 69.00 | 34.85 |     | 28.12 | 68.31 |     |
| --- | --- | ------ | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ----- | --- |
GTR
N.w/ExcluIR 61.44 83.82 88.34 71.01 65.64† 76.98† 39.05† 28.46 69.98†
|     |     | NQ320k |     |     | 60.08 | 84.19 | 89.41 | 70.50 | 57.01 | 70.88 | 20.02 |     | 15.26 | 59.97 |     |
| --- | --- | ------ | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ----- | --- |
ColBERT
N.w/ExcluIR 60.20 83.59 88.60 70.29 57.91 73.52† 19.30 13.05 59.71
|     |     | NQ320k |     |     | 56.25 | 71.21 | 74.00 | 62.80 | 31.63 | 37.63 | 11.44 |     | 10.15 | 58.65 |     |
| --- | --- | ------ | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ----- | --- |
GENRE
N.w/ExcluIR 55.15 70.00 72.85 61.55 65.67† 73.01† 41.19† 20.31† 70.48†
|     |     | NQ320k |     |     | 55.24 | 75.13 | 80.97 | 63.86 | 43.54 | 55.17 | 16.11 |     | 15.27 | 60.02 |     |
| --- | --- | ------ | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ----- | --- |
SEAL
N.w/ExcluIR 53.86 74.84 80.34 62.78 70.39† 78.40† 52.14† 43.25† 78.02†
|     |     | NQ320k |     |     | 60.41 | 76.10 | 80.19 | 67.18 | 31.46 | 38.95 | 15.87 |     | 16.81 | 56.84 |     |
| --- | --- | ------ | --- | --- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --- | ----- | ----- | --- |
NCI
N.w/ExcluIR 60.61 76.53 80.55 67.46 56.92† 64.67† 41.13† 39.92† 72.97†
Table3:TheresultsoftheimpactofaugmentingNQ320kwiththeExcluIRtrainingset.†indicatessignificantimprovements
withp-value<0.05.
from achieving ideal performance. This is attributed to the 4.2 Howdoesincorporatingourdatasetinto
fact that negative documents are erroneously retrieved and trainingdataaffecttheperformance?
rankedhigh,indicatingthatthesemodelsfailtocomprehend
Previousexperimentshavedemonstratedthatmodelstrained
theexclusionarynatureofqueries.
onHotpotQAandNQ320kperformunsatisfactorilyonEx-
Second,thediversityoftrainingdataimpactsthemodel’s cluIR. We believe that this is partly due to a lack of exclu-
abilitytocomprehendexclusionaryqueries.Ascanbeseen sionary queries in the training data. Therefore, in this sec-
| from Table | 1   | and 2, | the models |     | trained on | NQ320k | ex- |     |     |     |     |     |     |     |     |
| ---------- | --- | ------ | ---------- | --- | ---------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tion,weincorporatetheExcluIRtrainingsetintothetrain-
| hibit better | performance |     | on  | ExcluIR | than those | trained | on  |     |     |     |     |     |     |     |     |
| ------------ | ----------- | --- | --- | ------- | ---------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ingdatatoassessitsimpactonperformance.Theresultsof
| HotpotQA. | We  | consider | that | this is | because | the queries | in  |     |     |     |     |     |     |     |     |
| --------- | --- | -------- | ---- | ------- | ------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
augmentingNQ320kwiththeExcluIRtrainingsetarepre-
NQ320k are more diverse and contain more exclusionary sented in Table 3. Due to space limitations, the results of
queries. Therefore, increasing the domain and diversity of augmentingHotpotQAareincludedinappendix.Foreaseof
trainingdatacanbebeneficialforexclusionaryretrieval.To
analysis,wehavesummarizedtheresultsfrombothtablesin
| further | investigate | how | expanding |     | the training | data | influ- |     |     |     |     |     |     |     |     |
| ------- | ----------- | --- | --------- | --- | ------------ | ---- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
Figure4.Fromtheresults,wehavethreemainobservations.
| ences performance, |     | we  | conducted |     | additional | experimental |     |        |         |     |         |          |          |     |          |
| ------------------ | --- | --- | --------- | --- | ---------- | ------------ | --- | ------ | ------- | --- | ------- | -------- | -------- | --- | -------- |
|                    |     |     |           |     |            |              |     | First, | merging | the | ExcluIR | training | set into | the | training |
analyses. We have conducted further experimental analysis data can enhance most models’ ability to comprehend ex-
inSection4.2. clusionaryqueries.Additionally,theperformanceofallgen-
| Additionally, |     | as expected, |            | sparse | retrieval        |     | methods |         |           |           |             |     |                   |     |          |
| ------------- | --- | ------------ | ---------- | ------ | ---------------- | --- | ------- | ------- | --------- | --------- | ----------- | --- | ----------------- | --- | -------- |
|               |     |              |            |        |                  |     |         | erative | retrieval | models    | on ExcluIR  |     | has significantly |     | im-      |
| demonstrate   | a   | significant  | limitation |        | in comprehending |     | the     |         |           |           |             |     |                   |     |          |
|               |     |              |            |        |                  |     |         | proved. | For       | instance, | with NQ320k | as  | the original      |     | dataset, |
exclusionarynatureofqueries,sotheyhavealmostnoabil-
|               |          |     |          |     |          |        |        | SEAL | achieves | 18% improvement |     | (60.02% |     | vs. 78.02%) | in  |
| ------------- | -------- | --- | -------- | --- | -------- | ------ | ------ | ---- | -------- | --------------- | --- | ------- | --- | ----------- | --- |
| ity to handle | ExcluIR. |     | As shown | in  | Table 2, | the RR | scores |      |          |                 |     |         |     |             |     |
RRbyintegratingtheExcluIRtrainingset,withonlyasmall
of BM25 and DocT5Query are only 53.48% and 53.85%, (1.08%) decrease (63.86% vs. 62.78%) in performance on
| which are | only | slightly | higher | than | random. | Their | ∆R@1 |     |          |            |         |         |             |     |          |
| --------- | ---- | -------- | ------ | ---- | ------- | ----- | ---- | --- | -------- | ---------- | ------- | ------- | ----------- | --- | -------- |
|           |      |          |        |      |         |       |      | the | original | test data. | This is | because | the ExcluIR |     | training |
and∆MRRscoresarelowerthanmostneuralretrievalmod-
|             |            |     |      |       |          |         |         | set | contains | a large number | of  | exclusionary |     | queries, | which |
| ----------- | ---------- | --- | ---- | ----- | -------- | ------- | ------- | --- | -------- | -------------- | --- | ------------ | --- | -------- | ----- |
| els trained | on NQ320k. |     | This | is an | expected | result, | because |     |          |                |     |              |     |          |       |
canhelptheretrievalmodeltocomprehendtheexclusionary
thesemethodsarebasedonalexicalmatchbetweenqueries
natureofqueriesbetter.
anddocuments.Thislimitationpreventsthemfromfocusing Second,whentrainingdatacontainexclusionaryqueries,
ontheexclusionaryphrasesinthequery,insteadleadingto generativeretrievalmodelsarebetterathandlingexclusion-
ahighrelevancescorefornegativedocuments.
|              |     |         |          |     |             |     |          | ary   | retrieval | task compared      | to    | dense     | retrieval | models. | As      |
| ------------ | --- | ------- | -------- | --- | ----------- | --- | -------- | ----- | --------- | ------------------ | ----- | --------- | --------- | ------- | ------- |
| Furthermore, |     | we also | evaluate | the | performance |     | of addi- |       |           |                    |       |           |           |         |         |
|              |     |         |          |     |             |     |          | shown | in        | Figure 4, although | dense | retrieval |           | models  | trained |
tionalmodelstrainedondifferentdatasetsinExcluIR.Due
|     |     |     |     |     |     |     |     | on  | two original | datasets | perform | better | on  | ExcluIR, | aug- |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | ------- | ------ | --- | -------- | ---- |
tospacelimitations,theseresultsarepresentedinappendix2.
|     |     |     |     |     |     |     |     | menting | with | the ExcluIR | training | set | leads | to a | greater |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ---- | ----------- | -------- | --- | ----- | ---- | ------- |
improvementingenerativeretrievalmodels,ultimatelysur-
passingdenseretrievalmethodsoverall.Onaverage,genera-
2Appendixisavailableathttps://arxiv.org/abs/2404.17288 tiveretrievalmodelsachievea17.75%improvement,incon-
13299

|     |     |     |     |     |     | Trainingset | Model       |     | Base  | Large  |     |
| --- | --- | --- | --- | --- | --- | ----------- | ----------- | --- | ----- | ------ | --- |
|     |     |     |     |     |     |             | DPR         |     | 54.02 | 54.25↑ |     |
|     |     |     |     |     |     |             | Sentence-T5 |     | 55.41 | 53.78↓ |     |
HotpotQA
|     |     |     |     |     |     |           | GENRE       |     | 52.10 | 49.01↓ |     |
| --- | --- | --- | --- | --- | --- | --------- | ----------- | --- | ----- | ------ | --- |
|     |     |     |     |     |     |           | NCI         |     | 50.93 | 50.64↓ |     |
|     |     |     |     |     |     |           | DPR         |     | 61.19 | 62.63↑ |     |
|     |     |     |     |     |     | HotpotQA  | Sentence-T5 |     | 66.75 | 69.01↑ |     |
|     |     |     |     |     |     | w/ExcluIR | GENRE       |     | 69.07 | 70.96↑ |     |
|     |     |     |     |     |     |           | NCI         |     | 73.75 | 73.61↓ |     |
|     |     |     |     |     |     |           | DPR         |     | 58.76 | 61.62↑ |     |
|     |     |     |     |     |     |           | Sentence-T5 |     | 67.83 | 69.02↑ |     |
NQ320k
|     |     |     |     |     |     |           | GENRE       |     | 58.65 | 55.82↓ |     |
| --- | --- | --- | --- | --- | --- | --------- | ----------- | --- | ----- | ------ | --- |
|     |     |     |     |     |     |           | NCI         |     | 56.84 | 62.54↑ |     |
|     |     |     |     |     |     |           | DPR         |     | 61.00 | 63.47↑ |     |
|     |     |     |     |     |     | NQ320k    | Sentence-T5 |     | 68.00 | 69.65↑ |     |
|     |     |     |     |     |     | w/ExcluIR | GENRE       |     | 70.48 | 72.86↑ |     |
|     |     |     |     |     |     |           | NCI         |     | 72.97 | 74.45↑ |     |
Figure 4: Performance of models under different training Table 4: RR scores with different model sizes on ExcluIR.
data settings. The upper figures show the RR score of var- ↑ indicates that an increase in model size improves perfor-
mance,while↓indicatestheopposite.
| ious models | on              | the ExcluIR | benchmark,      | and the | lower fig- |     |     |     |     |     |     |
| ----------- | --------------- | ----------- | --------------- | ------- | ---------- | --- | --- | --- | --- | --- | --- |
| ures show   | the performance |             | of these models | on      | HotpotQA   |     |     |     |     |     |     |
andNQ320k.Thedifferentcolorsofthebarsrepresentdif-
ferenttrainingdata. uncased and bert-large-uncased. For sentence-t5, GENRE,
andNCI,weadoptt5-baseandt5-large.
TheresultsarepresentedinTable4.Wenotethatincreas-
trast to the average 4.77% improvement observed in dense ing the model size generally improves performance on Ex-
retrieval models. This is because the generative retrieval cluIRwhenthetrainingdataincludesexclusionaryqueries.
model is more suitable for capturing the complex relation- This isconsistentwith observationsby Ravichander, Gard-
ships between queries and documents in terms of model ner, and Marasovic´ (2022), who show that larger models
arebetteratunderstandingtheimplicationsofnegatedstate-
| architecture. | We  | present | a more detailed | analysis | in Sec- |                   |     |     |     |     |     |
| ------------- | --- | ------- | --------------- | -------- | ------- | ----------------- | --- | --- | --- | --- | --- |
| tion4.4.      |     |         |                 |          |         | mentsindocuments. |     |     |     |     |     |
Third, ColBERT fails to achieve satisfactory perfor- However, when training on original datasets, increasing
mance,evenafterfine-tuningonExcluIR.Amongthemod- the model size does not always lead to improved perfor-
elstrainedwiththeExcluIRtrainingset,ColBERTexhibits mance on ExcluIR. We conducted additional experiments
thelowestperformance.ThisisbecauseColBERTcalculates onmoremodels.Theresultsindicatedthattheperformance
|              |           |     |                |             |        | of stsb-roberta-large | decreases |     | compared | to stsb-roberta- |     |
| ------------ | --------- | --- | -------------- | ----------- | ------ | --------------------- | --------- | --- | -------- | ---------------- | --- |
| the document | relevance |     | score based on | token-level | match- |                       |           |     |          |                  |     |
ing, leading it to overlook exclusionary phrases in queries, base. This indicates that simply increasing model size can-
which is crucial for exclusionary retrieval. We have visual- notsolvethechallengesofexclusionaryretrieval,weshould
izedtherelevancecalculationofColBERTtofurtherunder- investigate building more training data and proposing new
| standitsperformanceinappendix. |     |          |              |         |             | trainingstrategies. |     |     |     |     |     |
| ------------------------------ | --- | -------- | ------------ | ------- | ----------- | ------------------- | --- | --- | --- | --- | --- |
| Moreover,                      | we  | consider | that a model | trained | only on our |                     |     |     |     |     |     |
4.4 Whyaregenerativeretrievalmodelssuperior
datasetwouldperformwellonExcluIRbutpoorlyonHot-
potQA and NQ320k. This is because the diversity of train- inExcluIR?
| ing data | is crucial | for training | a powerful  | retrieval | model.    |            |                  |      |          |            |     |
| -------- | ---------- | ------------ | ----------- | --------- | --------- | ---------- | ---------------- | ---- | -------- | ---------- | --- |
|          |            |              |             |           |           | Generative | retrieval models | have | inherent | advantages | in  |
| We have  | conducted  | preliminary  | experiments | on        | Sentence- |            |                  |      |          |            |     |
comprehendingexclusionaryqueries.Wetrytoanalyzeand
T5 to confirm this, due to space limitations, the results are explain the reason based on the architecture of generative
| presented | in appendix. |     | The results indicate | that | the model | models. |     |     |     |     |     |
| --------- | ------------ | --- | -------------------- | ---- | --------- | ------- | --- | --- | --- | --- | --- |
trained only on our dataset struggles to perform well on First, as a comparison, we show that bi-encoder models
standardretrievaldatasetsduetothelackofgeneraltraining have a representation bottleneck for exclusionary queries.
data,sowedidn’tconductmoreexperimentsinthissetting.
Whentwodocumentsaresimilarbuthavesomedifferences
thattheuserwouldliketodistinguish,itisdifficulttoensure
4.3 Howdoesmodelsizeaffectperformance?
|     |     |     |     |     |     | that the vector | representation | of  | the query | remains | distant |
| --- | --- | --- | --- | --- | --- | --------------- | -------------- | --- | --------- | ------- | ------- |
To analyze the impact of model size on the performance fromthenegativedocumentwhilecloselyaligningwiththe
of ExcluIR, we increase model sizes of DPR, sentence-t5, positive document. This representation bottleneck prevents
GENRE, and NCI, and train them on different datasets. the model from correctly comprehending the true intent of
Specifically, for DPR, we use two variants: bert-base- thequery.Wepresentthisproofinappendix.
13300

Dense Retrieval  time-prohibitive, which is why we excluded them from the
Maximal inner product search
mainexperiments.
| Dual Encoder |     | Docs |     | Query | Dual Encoder |     |     |     |     |     |     |     |     |
| ------------ | --- | ---- | --- | ----- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
5 RelatedWork
Cross Attention in Generative Retrieval Models Decoder
|     |     |     |     |     |     |     | Early studies | in  | exclusionary |             | retrieval primarily |           | focus on |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------------ | ----------- | ------------------- | --------- | -------- |
|     |     |     |     |     |     |     | keyword-based |     | methods.     | These       | approaches          | typically | treat    |
|     |     |     |     |     |     |     | user queries  | as  | logical      | expressions | of                  | boolean   | opera-   |
tions(NakkouziandEastman1990;Strzalkowski1995;Mc-
|     |     |     |     |     |     |     | Quire and     | Eastman | 1998; | Harvey      | et al.            | 2003). | However, |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ------- | ----- | ----------- | ----------------- | ------ | -------- |
|     |     |     |     |     |     |     | these methods | depend  |       | on explicit | and deterministic |        | rules,   |
query token vector lack the flexibility to handle subtle exclusions, and are not
Exclusionary Cross Attention suitableformorerealisticretrievalscenarios.
exclusionary term vector,
like 'EXCEPT' Phrase Weight In addition, there is a task related to exclusionary re-
trieval,knownasargumentretrieval(Wachsmuth,Syed,and
docid token vector
Stein2018),whichaimstoretrievethebestcounterargument
Figure5:Summaryoftheanalysisthatshowsthedifferences for a given argument on any controversial topic. While ar-
between dense retrieval and generative retrieval models in gument retrieval implicitly requires the model to find the
handlingExcluIR. counterargument to the query, the intention of exclusion is
|     |     |     |     |     |     |     | not explicitly    | expressed |              | in the    | query. Wang | et            | al. (2022b) |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --------- | ------------ | --------- | ----------- | ------------- | ----------- |
|     |     |     |     |     |     |     | first investigate |           | exclusionary | retrieval | in          | Text-to-Video | Re-         |
Generativeretrievalmodelsadoptasequence-to-sequence
|     |     |     |     |     |     |     | trieval (T2VR). |     | They | demonstrate | that | existing | video re- |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ---- | ----------- | ---- | -------- | --------- |
framework,suchasT5orBART,whichestimatestheprob- trievalmodelsperformedpoorlywhendealingwithqueries
ability of generating the document IDs given the query us- like “find shots of kids sitting on the floor and not playing
ing a conditional probability model: P(d|q). When gener- withthedog.”Tothebestofourknowledge,therehasbeen
ating document IDs, multiple cross-attention layers in the noresearchonexclusionaryretrievalindocumentretrieval.
decodercancapturethetoken-levelsemanticinformationin
(Weller,Lawrie,andVanDurme2024)introduceNevIR,
thequery,aphenomenonalsoexploredbyWuetal.(2024).
|     |     |     |     |     |     |     | a benchmark | designed |     | to assess | the ability | of neural | infor- |
| --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | --- | --------- | ----------- | --------- | ------ |
AssumingthedecoderconsistsofLlayers,forthel-thlayer mationretrievalsystemstohandlenegation.NevIRrequires
(0≤l<L),thecross-attentionlayerisgivenby: retrieval models to rank two documents that differ only in
(cid:18) Q(l)K(l)T(cid:19) negation, where both documents remain consistent in all
|     | S(l+1) | =softmax |     | √   | V(l), | (2) |               |        |     |               |            |     |         |
| --- | ------ | -------- | --- | --- | ----- | --- | ------------- | ------ | --- | ------------- | ---------- | --- | ------- |
|     |        |          |     |     |       |     | other aspects | except | the | key negation. | Similarly, |     | Rokach, |
d
|     |     |     |     | k   |     |     | Romano,andMaimon(2008);Koopmanetal.(2010)inves- |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- |
where Q(l) = W(l)S(l), K(l) = W(l)H(l), V(l) = tigatetheimpactofnegationcontextswithindocumentson
|     |     | q   |     |     | k q |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
W(l)H(l), H(l) retrievalperformance.Forexample,asearchfor“headache”
|     | and | =   | [e ,··· | ,e ] are | query token | vec- |     |     |     |     |     |     |     |
| --- | --- | --- | ------- | -------- | ----------- | ---- | --- | --- | --- | --- | --- | --- | --- |
v q q q1 qN mightretrievepatientrecordscontaining“thepatienthasno
torsgeneratedbyencoder,S(l)
|                                                 |     |     | =[e | d1 ,··· | ,e dM ]aregener- |      |                                                        |               |     |     |                   |     |           |
| ----------------------------------------------- | --- | --- | --- | ------- | ---------------- | ---- | ------------------------------------------------------ | ------------- | --- | --- | ----------------- | --- | --------- |
|                                                 |     |     |     |         |                  |      | symptoms                                               | of headache.” |     | Our | work is different |     | as we fo- |
| atedembeddingvectorsfordocidtokensatl-thlayer,W |     |     |     |         |                  | (l), |                                                        |               |     |     |                   |     |           |
|                                                 |     |     |     |         |                  | q    | cusonexclusionaryretrieval,studyingwhethertheretrieval |               |     |     |                   |     |           |
W(l) and W(l) are learnable cross-attention weight matri- modelcancomprehendtheintentofexclusionaryqueries.
| k            |           | v                   |               |              |          |           |     |     |     |            |     |     |     |
| ------------ | --------- | ------------------- | ------------- | ------------ | -------- | --------- | --- | --- | --- | ---------- | --- | --- | --- |
| ces. We      | visualize | the cross-attention |               | architecture |          | in gener- |     |     |     |            |     |     |     |
| ative models |           | to summarize        | our analysis. |              | As shown | in Fig-   |     |     |     |            |     |     |     |
|              |           |                     |               |              |          |           |     |     | 6   | Conclusion |     |     |     |
ure5,themulti-levelcross-attentionmechanismallowsthe
| model | to strongly | focus | on key | terms in | the query, | includ- |     |     |     |     |     |     |     |
| ----- | ----------- | ----- | ------ | -------- | ---------- | ------- | --- | --- | --- | --- | --- | --- | --- |
ingexclusionaryphrases(highlightedindarkgreen).Thus, In this work, we focus on a common yet understudied re-
when faced with queries with complex semantics, genera- trieval scenario called exclusionary retrieval, where users
tiveretrievalmodelsarecapableofeffectivelycapturingthe explicitly express which information they do not want
queryintent. to obtain. We have provided the community with a new
Notably, this architectural advantage is also present in benchmark,namedExcluIR,whichfocusesonexclusionary
cross-encoders,suchastheclassicBERTre-ranker.Wehave queries that explicitly express the information users do not
evaluated the performance of cross-encoder models on Ex- want to obtain. We have conducted extensive experiments
cluIR, the results are presented in appendix. It can be seen thatdemonstratethatexistingretrievalmethodswithdiffer-
that, within the zero-shot setting, a strong cross-encoder ent architectures perform poorly on ExcluIR. Notably, Ex-
model outperforms both dense retrieval and generative re- cluIR cannot be solved by simply adding training data do-
trieval models on ExcluIR. This result is expected, as the mainsorincreasingmodelsizes.Additionally,ouranalyses
cross-encodercalculatesthesimilaritybetweenaqueryand indicatethatgenerativeretrievalmodelsinherentlyexcelat
adocumentindividually,allowingittobetterunderstandthe comprehending exclusionary queries compared with sparse
relationbetweenthequeryandthedocument.However,em- and dense retrieval models. We hope that this work can in-
ploying such models for retrieval from the entire corpus is spirefutureresearchonExcluIR.
13301

Acknowledgements McQuire,A.R.;andEastman,C.M.1998.Theambiguityof
negationinnaturallanguagequeriestoinformationretrieval
| This work | was | supported | by the Key | R&D | Program | of  |     |     |     |     |     |     |     |
| --------- | --- | --------- | ---------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
ShandongProvincewithgrant2024CXGC010108,theNat- systems. Journal of the American Society for Information
Science,49(8):686–692.
| ural Science | Foundation |     | of China (62472261, |     | 62102234, |     |     |     |     |     |     |     |     |
| ------------ | ---------- | --- | ------------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
62372275, 62272274, 62202271, T2293773, 62072279), Nakkouzi,Z.S.;andEastman,C.M.1990. Queryformula-
the National Key R&D Program of China with grant tionforhandlingnegationininformationretrievalsystems.
No.2022YFC3303004, the Natural Science Foundation of Journal of the American Society for Information Science,
| ShandongProvince(ZR2021QF129),andbytheDutchRe- |     |     |     |     |     |     | 41(3):171–182. |     |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
searchCouncil(NWO),underprojectnumbers024.004.022,
|     |     |     |     |     |     |     | Ni, J.; Abrego, |     | G. H.; | Constant, | N.; | Ma, J.; | Hall, K.; Cer, |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ------ | --------- | --- | ------- | -------------- |
NWA.1389.20.183, and KICH3.LTP.20.006, and the Euro- D.; and Yang, Y. 2022a. Sentence-T5: Scalable Sentence
pean Union’s Horizon Europe program under grant agree- Encoders from Pre-trained Text-to-Text Models. In Find-
ment No 101070212. All content represents the opinion of ingsoftheAssociationforComputationalLinguistics:ACL
theauthors,whichisnotnecessarilysharedorendorsedby 2022,1864–1874.
theirrespectiveemployersand/orsponsors.
Ni,J.;Qu,C.;Lu,J.;Dai,Z.;Abrego,G.H.;Ma,J.;Zhao,
|     |     |     |     |     |     |     | V.; Luan, | Y.; Hall, | K.; | Chang, | M.-W.; | et al. 2022b. | Large |
| --- | --- | --- | --- | --- | --- | --- | --------- | --------- | --- | ------ | ------ | ------------- | ----- |
References
|     |     |     |     |     |     |     | Dual Encoders | Are | Generalizable |     | Retrievers. |     | In Proceed- |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------------- | --- | ----------- | --- | ----------- |
Bevilacqua,M.;Ottaviano,G.;Lewis,P.;Yih,S.;Riedel,S.; ingsofthe2022ConferenceonEmpiricalMethodsinNatu-
andPetroni,F.2022. Autoregressivesearchengines:Gener- ralLanguageProcessing,9844–9855.
atingsubstringsasdocumentidentifiers.AdvancesinNeural
|     |     |     |     |     |     |     | Nogueira, | R.; Lin, | J.; | and | Epistemic, | A. 2019. | From |
| --- | --- | --- | --- | --- | --- | --- | --------- | -------- | --- | --- | ---------- | -------- | ---- |
InformationProcessingSystems,35:31668–31683. doc2querytodocTTTTTquery. Onlinepreprint,6:2.
| Cherry, K. | 2020. | How | we use selective | attention |     | to filter |             |          |     |          |     |          |             |
| ---------- | ----- | --- | ---------------- | --------- | --- | --------- | ----------- | -------- | --- | -------- | --- | -------- | ----------- |
|            |       |     |                  |           |     |           | Raffel, C.; | Shazeer, | N.; | Roberts, | A.; | Lee, K.; | Narang, S.; |
informationandfocus. VerywellMind. Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2020. Explor-
De Cao, N.; Izacard, G.; Riedel, S.; and Petroni, F. 2020. ingthelimitsoftransferlearningwithaunifiedtext-to-text
ICLR 2021-9th Inter- transformer. The Journal of Machine Learning Research,
| Autoregressive | Entity | Retrieval. | In  |     |     |     |     |     |     |     |     |     |     |
| -------------- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
21(1):5485–5551.
| national | Conference | on  | Learning Representations, |     |     | volume |     |     |     |     |     |     |     |
| -------- | ---------- | --- | ------------------------- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
2021.ICLR. Ravichander, A.; Gardner, M.; and Marasovic´, A. 2022.
|     |     |     |     |     |     |     | CONDAQA: | A   | Contrastive |     | Reading | Comprehension |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | --- | ----------- | --- | ------- | ------------- | --- |
Devlin,J.;Chang,M.-W.;Lee,K.;andToutanova,K.2019.
|     |     |     |     |     |     |     | Dataset | for Reasoning |     | about | Negation. | In  | Proceedings |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------------- | --- | ----- | --------- | --- | ----------- |
BERT:Pre-trainingofDeepBidirectionalTransformersfor
Language Understanding. In Proceedings of NAACL-HLT, of the 2022 Conference on Empirical Methods in Natural
| 4171–4186. |     |     |     |     |     |     | LanguageProcessing,8729–8755. |     |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | --- | --- | --- |
Harvey,V.J.;Baugh,J.M.;Johnston,B.A.;Ruzich,C.M.; Robertson, S.; and Zaragoza, H. 2009. The probabilistic
Grant, A. J.; et al. 2003. The challenge of negation in relevanceframework:BM25andbeyond. Foundationsand
searches and queries. Review of Business Information Sys- TrendsinInformationRetrieval,3(4):333–389.
tems(RBIS),7(4):63–76. Robertson,S.E.;andWalker,S.1997.Onrelevanceweights
Karpukhin,V.;Oguz,B.;Min,S.;Lewis,P.;Wu,L.;Edunov, withlittlerelevanceinformation. InProceedingsofthe20th
|                              |     |     |                       |     |     |     | annual | international | ACM | SIGIR | conference |     | on Research |
| ---------------------------- | --- | --- | --------------------- | --- | --- | --- | ------ | ------------- | --- | ----- | ---------- | --- | ----------- |
| S.;Chen,D.;andYih,W.-t.2020. |     |     | DensePassageRetrieval |     |     |     |        |               |     |       |            |     |             |
anddevelopmentininformationretrieval,16–24.
| for Open-Domain |     | Question | Answering. | In  | Proceedings | of  |     |     |     |     |     |     |     |
| --------------- | --- | -------- | ---------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
the2020ConferenceonEmpiricalMethodsinNaturalLan- Rokach, L.; Romano, R.; and Maimon, O. 2008. Negation
guageProcessing(EMNLP),6769–6781. recognition in medical narrative reports. Information Re-
trieval,11:499–538.
| Khattab,O.;andZaharia,M.2020. |     |     | ColBERT:Efficientand |     |     |     |     |     |     |     |     |     |     |
| ----------------------------- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
effective passage search via contextualized late interaction Strzalkowski, T. 1995. Natural language information re-
overBERT. InProceedingsofthe43rdInternationalACM trieval.InformationProcessing&Management,31(3):397–
417.
SIGIRconferenceonresearchanddevelopmentinInforma-
tionRetrieval,39–48. Treisman, A. M. 1964. Selective attention in man. British
MedicalBulletin,20(1):12–16.
| Koopman, | B.; Bruza, | P.; | Sitbon, L.; | and Lawley, | M.  | 2010. |     |     |     |     |     |     |     |
| -------- | ---------- | --- | ----------- | ----------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
Analysis of the effect of negation on information retrieval Wachsmuth, H.; Syed, S.; and Stein, B. 2018. Retrieval
ofmedicaldata. InProceedingsof15thAustralasianDoc- ofthebestcounterargumentwithoutpriortopicknowledge.
ument Computing Symposium, 89–92. School of Computer In Proceedings of the 56th Annual Meeting of the Associa-
ScienceandIT,RMITUniversity. tionforComputationalLinguistics(Volume1:LongPapers),
| Kwiatkowski, | T.; | Palomaki, | J.; Redfield, | O.; | Collins, | M.; | 241–251. |     |     |     |     |     |     |
| ------------ | --- | --------- | ------------- | --- | -------- | --- | -------- | --- | --- | --- | --- | --- | --- |
Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin, Wang, Y.; Hou, Y.; Wang, H.; Miao, Z.; Wu, S.; Chen, Q.;
| J.;Lee,K.;etal.2019. |     | Naturalquestions:abenchmarkfor |     |     |     |     |          |          |       |          |        |            |          |
| -------------------- | --- | ------------------------------ | --- | --- | --- | --- | -------- | -------- | ----- | -------- | ------ | ---------- | -------- |
|                      |     |                                |     |     |     |     | Xia, Y.; | Chi, C.; | Zhao, | G.; Liu, | Z.; et | al. 2022a. | A neural |
question answering research. Transactions of the Associa- corpusindexerfordocumentretrieval. AdvancesinNeural
tionforComputationalLinguistics,7:453–466. InformationProcessingSystems,35:25600–25614.
LaBerge, D. L. 1990. Attention. Psychological Science, Wang, Z.; Chen, A.; Hu, F.; and Li, X. 2022b. Learn to
1(3):156–162. understandnegationinvideoretrieval.InProceedingsofthe
13302

30th ACM International Conference on Multimedia, 434–
443.
Weller, O.; Lawrie, D.; and Van Durme, B. 2024. NevIR:
Negation in Neural Information Retrieval. In Graham, Y.;
and Purver, M., eds., Proceedings of the 18th Conference
of the European Chapter of the Association for Computa-
tionalLinguistics(Volume1:LongPapers),2274–2287.St.
Julian’s,Malta:AssociationforComputationalLinguistics.
Wu, S.; Wei, W.; Zhang, M.; Chen, Z.; Ma, J.; Ren,
Z.; de Rijke, M.; and Ren, P. 2024. Generative Re-
trieval as Multi-Vector Dense Retrieval. arXiv preprint
arXiv:2404.00684.
Yang,Z.;Qi,P.;Zhang,S.;Bengio,Y.;Cohen,W.;Salakhut-
dinov,R.;andManning,C.D.2018. HotpotQA:ADataset
forDiverse,ExplainableMulti-hopQuestionAnswering. In
Proceedingsofthe2018ConferenceonEmpiricalMethods
inNaturalLanguageProcessing,2369–2380.
13303
