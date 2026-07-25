PublishedasaconferencepaperatICLR2025
| BEYOND    |     | CONTENT   |     | RELEVANCE: |           |     | EVALUATING |        |     | IN- |
| --------- | --- | --------- | --- | ---------- | --------- | --- | ---------- | ------ | --- | --- |
|           |     | FOLLOWING |     |            | RETRIEVAL |     |            | MODELS |     |     |
| STRUCTION |     |           |     | IN         |           |     |            |        |     |     |
JianqunZhou1∗,YuanleiZheng4‡,WeiChen4∗
QianqianZheng1,HuiSu2,WeiZhang1,RuiMeng3†,XiaoyuShen1,5†
1NingboKeyLaboratoryofSpatialIntelligenceandDigitalDerivative,InstituteofDigitalTwin,
| EasternInstituteofTechnology,Ningbo |     |     |     | 2MeituanInc. |     | 3SalesforceResearch |     |     |     |     |
| ----------------------------------- | --- | --- | --- | ------------ | --- | ------------------- | --- | --- | --- | --- |
4SchoolofSoftwareEngineering,HuazhongUniversityofScienceandTechnology
5EngineeringResearchCenterofChipletDesignandManufacturingofZhejiangProvince
| ruimeng@salesforce.com, |     |     |     | xyshen@eitech.edu.cn |     |     |     |     |     |     |
| ----------------------- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- |
5202 raM 5  ]RI.sc[  2v14832.0142:viXra
ABSTRACT
Instruction-followingcapabilitiesinLLMshaveprogressedsignificantly,enabling
|     | more | complex user | interactions | through | detailed | prompts. |     | However, | retrieval |     |
| --- | ---- | ------------ | ------------ | ------- | -------- | -------- | --- | -------- | --------- | --- |
systemshavenotmatchedtheseadvances,mostofthemstillreliesontraditional
|     | lexical | and semantic | matching | techniques | that | fail to | fully | capture | user intent. |     |
| --- | ------- | ------------ | -------- | ---------- | ---- | ------- | ----- | ------- | ------------ | --- |
Recenteffortshaveintroducedinstruction-awareretrievalmodels,butthesepri-
|     | marily                                                   | focus on | intrinsic | content relevance, |     | which | neglects | the importance     | of  |     |
| --- | -------------------------------------------------------- | -------- | --------- | ------------------ | --- | ----- | -------- | ------------------ | --- | --- |
|     | customizedpreferencesforbroaderdocument-levelattributes. |          |           |                    |     |       |          | Thisstudyevaluates |     |     |
theinstruction-followingcapabilitiesofvariousretrievalmodelsbeyondcontent
|     | relevance,includingLLM-baseddenseretrievalandrerankingmodels. |     |     |     |     |     |     |     | Wedevelop |     |
| --- | ------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --------- | --- |
InfoSearch,anovelretrievalevaluationbenchmarkspanningsixdocument-level
|     | attributes: | Audience,Keyword,Format,Language,Length,andSource,andintro- |     |     |     |     |     |     |     |     |
| --- | ----------- | ----------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
ducenovelmetrics–StrictInstructionComplianceRatio(SICR)andWeighted
InstructionSensitivityEvaluation(WISE)toaccuratelyassessthemodels’respon-
|     | sivenesstoinstructions. |     | Ourfindingsindicatethatalthoughfine-tuningmodelson |     |     |     |     |     |     |     |
| --- | ----------------------- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
instruction-awareretrievaldatasetsandincreasingmodelsizeenhanceperformance,
mostmodelsstillfallshortofinstructioncompliance.1
1 INTRODUCTION
Theadventofinstruction-followinginlargelanguagemodels(LLMs)hasgreatlyexpandedtheir
generativecapabilities(Brown,2020;Louetal.,2023), allowinguserstoexpressmorecomplex
intentionsthroughdetailedinstructions(Blacketal.,2022;Touvronetal.,2023). However,retrieval
systems have not kept pace with these advancements, continuing to rely on traditional lexical or
semanticmatchingtechniques(Xiongetal.,2020;Wangetal.,2022;Xiaoetal.,2023). Asaresult,
whileusershavegrownaccustomedtointeractingwithgenerativemodelsusingintricateinstructions
(Teametal.,2023;Baietal.,2023;Achiametal.,2023),theirretrievalbehaviorremainslimitedto
keyword-basedqueriesfollowedbymanualfilteringofresultstofindthedesiredinformation. Several
studieshavestartedtoexploreinstruction-awareretrieversthatcaninteractwithusersasseamlessly
asgenerativemodels,buttheseprimarilyfocusontask-levelinstructions(Asaietal.,2023;Wang
et al., 2023; Peng et al., 2024), guiding retrievers with one instruction for each task. While this
task-levelinstructionisessentialforadaptingasingleretrievalmodeltomultiplepredefinedscenarios,
itfallsshortofmeetingusers’customizeddemandsbeyondstandardtasks(Welleretal.,2024b).
Recentworkshaveshiftedfromtask-leveltoinstance-levelinstructions,providingtailoredinstructions
foreachinstancetobetteralignwithcustomizedneeds(Welleretal.,2024a;Ohetal.,2024). These
approachesspecifyinstructionsthatwhichcontenttoincludeorexclude,thusclarifyinguserintent.
While they greatly enrich the diversity of instructions, their primary focus remains on content
relevance. When searching for certain documents, users typically care about two aspects: the
informationalcontentanditspresentation(Taylor,1962;Mizzaro,1998),includingdocument-level
| ∗Authorscontributedequally.† |     |     | Correspondingauthors. |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --------------------- | --- | --- | --- | --- | --- | --- | --- |
1Wereleaseourdatasetandcodeonhttps://github.com/EIT-NLP/InfoSearch

PublishedasaconferencepaperatICLR2025
attributes such as length, language, and format. A sole focus on content relevance neglects the
importanceofcustomizedpreferencesforbroaderdocument-levelattributes.
We believe that a truly instruction-aware retrieval system must go beyond content relevance to
accommodateavarietyofuser-defineddocumentattributes. Tofurtherresearchinthisdirection,
weproposeInfoSearch,anovelbenchmarkdesignedtoevaluateIRmodelsbasedontheirability
tofollowcustomizedinstructionsacrosssixstructureddimensions: Audience,Keyword,Format,
Language, Length, and Source. These dimensions encompass key document-level features that
addressuserneedsbeyondcontentrelevance. Additionally,weincludebothinstructedandreverse-
instructedmodestoassessthemodel’sabilitytocomprehendinstructionsinbothaffirmativeand
negativeformats. Eachinstructioniscarefullycraftedandmanuallyvalidatedtoensurenaturalness
andrepresentativenessofcomplexreal-worldscenarios.
Beyond the comprehensiveness of datasets, well-defined evaluation metrics are essential to thor-
oughlyassesstheinstruction-followingcapabilitiesofretrievalmodels. WhiletraditionalIRmetrics
like nDCG and MRR are primarily effective for assessing content relevance in ad-hoc retrieval
tasks(Welleretal.,2024a),weproposenewmetricsspecificallydesignedtomeasurethedepthof
instruction-following capabilities in retrieval models. By structuring the evaluation across these
separatedimensionsandmodes,weofferadetailedanalysisofhowwellmodelsfollowinstructions
oneachcondition,providingclearerinsightsintotheirstrengthsandlimitations. Overall,fine-tuning
oninstruction-awareretrievaldatasetsandincreasingmodelparametersimproveinstruction-following
capabilities,withre-rankingmodelsoutperformingretrievalmodelsinthisaspect. However,both
approaches still show considerable room for improvement in meeting the standards set by our
benchmark.
Ourcontributionscanbesummarizedasfollows:
• WeproposeInfoSearch,anevaluationframeworkthatcoverssixkeydimensions: Audience,
Keyword,Format,Language,Length,andSource,toassessretrievalmodels’abilitytofollow
complexinstructionsbeyondcontentmatching.
• Weintroducetwonovelmetrics–StrictInstructionComplianceRatio(SICR)andWeighted
InstructionSensitivityEvaluation(WISE)–thatprovideamorenuancedandaccurateassessment
ofretrievalmodels’instructionadherencecomparedtotraditionalIRmetrics.
• Weevaluate15retrievalmodels,encompassing1sparseretrievalmodel,8bi-encoder-based
densemodels,and6LLM-basedrerankingmodels. Thisthoroughevaluationenablesacom-
prehensive comparison across diverse methodologies, delivering valuable insights into their
instruction-followingeffectiveness.
2 INFOSEARCH: CONSTRUCTION AND EVALUATION
Weconstructabenchmark,Instruction-FollowingSearch(InfoSearch),toevaluatingthesearch
models’abilitytofollowinstructions. InfoSearchiscomposedofquery-docpairsacrosssixdimen-
sionsandtwonovelmetricstomeasuremodels’responsivenesstoinstructions. Inthissection,§2.1
detailsthedimensionsettingsandretrievalmodesundertheInfoSerachframework,§2.2explainsthe
datasetconstructionprocess,and§2.3describesthedesignlogicbehindourproposedmetrics.
2.1 DATASETFRAMEWORK
Inreal-worldscenarios,usersexhibitawiderangeofcomplexanddiversesearchintentions. Theuse
oftailoredinstructionscanlinkthespecificsearchquerycontenttotherequirementsandpreferences
oftheuser. Instructionstypicallycontaindetailedinformationordocument-levelcharacteristicsthat
alignwithuserneeds,aimingtoenhancetheprecisionandrelevanceofsearchresults. Asshown
in Figure 1, we conducted an extensive analysis of real users and their underlying intentions to
identifysixfactors(dimensions)influencingsearchbehaviors: userbackground(Audience),specific
searchtermsortopics(Keyword),preferredformatforinformationpresentation(Format),required
responselength(Length),languagerequirement(Language),andinformationorigin(Source). To
enhancethediversityofinstructionsacrossthesesixdimensions,weestablishedmultipleconditional
brancheswithineachdimension,allowinginstructionstodynamicallyadaptandexpandbasedon
differentconditions. Moreover,evenwithinthesamedimensionandconditions,wecreatevaried

PublishedasaconferencepaperatICLR2025
Dimension Condition Retrieval Mode
Audience [Layman], [Expert] Original Mode:
“How can I reduce stress? Retrieve relevant passages that
Keyword [Keyword] answer the query. ”
Format [Post], [Code], [Manual] Instructed Mode:
“Tell me effective ways to reduce stress. Please provide the
Language [Chinese], [English] answer in English.”
Length [Sentence], [Paragraph], [Article] Reversely Instructed Mode:
“Tell me effective ways to reduce stress. Please do not
Source [Blog], [Forum], [News]
provide answers in English.”
Figure1: InfoSearchconsistsofsixdimensions,eachrepresentingadocument-levelfeaturewith
valuesdrawnfrompredefinedconditions. Queriesarepairedwithonedimensionandevaluatedin
threeretrievalmodesbasedonthegiveninstructions.
instructionsusingdiversewordingandexpressions. Thisapproachnotonlyenrichesthedatasetbut
alsostrengthenstherobustnessandreliabilityoftheevaluationframeworkbysimulatingabroad
rangeofpotentialuserinputs.
Drawinginspirationfrom(Zhangetal.,2024),weincorporatesemanticnegationintothedatasetby
reversingthemeaningofinstructionsacrosseachdimension. Thisapproachallowseachqueryto
beassociatedwithmultipleinstructions,coveringvariousconditionsandofferingbothpositiveand
negativesemanticcontexts. Thisensuresthatthemodelisexposedtothreedistinctretrievalmodes:
• OriginalMode: Thismodeservesasabaselinethatevaluatesthemodel’sbasicretrievalability
tofindpertinentinformationwithoutanyspecificconstraints.
• InstructedMode:Inthismode,themodelisrequiredtofinddocumentsthatarecontentrelevant
andsatisfytheconditionspecifiedintheinstruction.
• ReverselyInstructedMode: Inthismode,themodelisrequiredtofinddocumentsthatare
contentrelevantanddonot satisfytheconditionspecifiedintheinstruction, whichteststhe
model’sabilitytounderstandnegation.
By integrating six dimensions and three distinct retrieval modes, we have developed the com-
prehensiveevaluationdatasetInfoSearch. Thisdatasetservesasarobusttoolforsystematically
systematicallyevaluatingmodel’sabilitytointerpretandrespondaccuratelytodiverseinstructions
duringretrieval.
2.2 CONSTRUCTIONPROCESS
The primary objective of developing InfoSearch is to bridge queries with diverse instructions,
ensuring precise alignment between instructions and their corresponding target documents. We
achievethisbycollectingQuestion-Answer(Q-A)pairsforeachdimensionandexpandingthetarget
document pool through web-retrieved content. Figure 2 outlines the 7-step construction process.
Datasources,methodologicaldetails(e.g.,GPT-4prompts),implementationchallengesanddataset
statisticsareprovidedinAppendixA.
Step1: ConditionDetermination:Queriesarediversifiedviamultipleconditions,enablingasingle
querytoyielddistinctrelevantdocumentsdependingoncontextualrequirements.
Step2: Data Collection: To ensure query naturalness and minimize generation costs, we con-
sciouslyintegrateconditionswhenfilteringQ-Apairsfromexistingdatasetsorwebpages.
Step3: InstructionGeneration: Requiresproducingprecise,concise,andnaturalinstructionsthat
reflectnaturalconversationalpatterns,aligningwithusers’tendencytoexpressintents.
Step4: DocumentRewriting: Whenqueriesordocumentsinadequatelyaddressinstructionre-
quirements,GPT-4refinesexistingcontenttoproducecontextuallyappropriatedocuments.
Step5: Instruction Reversal: To verify whether ranking improvements stem from instruction
comprehension,instructionsemanticsaresystematicallyreversed,testingmodelrobustness
againstpersistenthigh-rankingresults.

PublishedasaconferencepaperatICLR2025
Step 2: Data Collection Step 3: Instruction Generation Step 5: Instruction Reversal Step 6: Hard Negative Generation
| Core Query: Advances in  |     | Blog |     | Blog |     |     |     |
| ------------------------ | --- | ---- | --- | ---- | --- | --- | --- |
artificial intelligence Ins 1: What are the recent  Reversed  Ins  1:  How  are  Hard Neg 1: In recent years, the
|     |     | Naretwifisc Aiarlt icle |     | Nreecwesn tA rticldeevelopments  |     |     |     |
| --- | --- | ----------------------- | --- | -------------------------------- | --- | --- | --- |
Doc  1:  This  blog  will  intelligence  Step 4:  in  art world has been buzzing with
|                                          |     | Inads vaNn:c eHmoewnt s?a re rPecleeanste   |           | Raervtiefircsieadl  Iinnste Nlli:g Wenhcaet?  arPel ethaese   |     | Hexacridt emNeengt  Nov: er( CtNheN  inNteErsWecSt)io--n  |     |
| ---------------------------------------- | --- | ------------------------------------------- | --------- | ------------------------------------------------------------- | --- | --------------------------------------------------------- | --- |
| Deoxcp loNre:  Scietnhcee  Newlaste--st  |     |                                             | Document  |                                                               |     |                                                           |     |
Baidov-iannscpeimreedn tcsa mine ararsti faicnida l  dpervoevloidpem ae nbtlso gin  daisrtciufiscsiianlg   Rewriting reacveonidt  parortvifidicei a lb loign tdeilslciguesnscineg   Aodf vancrceeast iivni tayr tifaicnida l intetcehllnigoelongcey .
|                                                  |     | inthteelsleig deenvceel?o pPmleeanstes .s hare  |     | atdhveasnec deemveenlotsp?m enPtsle. ase  |     | hAavrtei ficsipaal riknetedl ligceonnccee rhnass  maamdoen…g  |     |
| ------------------------------------------------ | --- | ----------------------------------------------- | --- | ----------------------------------------- | --- | ------------------------------------------------------------- | --- |
| AinI tehlleiglpe ncder iavnedrs  mdaectheicnte   |     |                                                 |     |                                           |     | not                                                           |     |
pleedaernstirnigan …s and obstacles  a news article covering  share  news  article  covering  privacy  advocates.  A  recent
faster ... these advancements? these advancements? survey conducted by the …
Step 7: Manual Review
Instruction-Following Search --- InfoSearch
|     | Figure2: | OverviewofthedatasetconstructionprocessforInfoSearch. |     |     |     |     |     |
| --- | -------- | ----------------------------------------------------- | --- | --- | --- | --- | --- |
Step6: HardNegativeGenerations:Adversarialexamplesareaddedtoresistthemodel’stendency
todependonsuperficialdocumentfeaturesratherthanquery-documentrelationships.
Step7: ManualReview: Anomalousoutputswereexcluded,prioritizingdocumentsthatconsis-
tentlyunderperformedorhadlowrelevancescores,followedbyexpertverification.
Byapplyingthedataconstructionprocessdescribedabove,theInfoSearchbenchmarkcomprises
600corequeries,1,598instructedqueries,1,598reverselyinstructedqueries,and6,392documents.
2.3 EVALUATIONMETRICS
Inreal-worldsearchsystems,userexperiencehingesontherelevanceoftop-Kresults,whichdirectly
reflectsmodelefficacy. Thus,instruction-followingmodelsmustbeevaluatedbasedonbothoriginal
query rankings and their responsiveness to instructions. While metrics like Robustness@k (Oh
et al., 2024) and p-MRR (Weller et al., 2024a) assess instruction compliance, they exhibit five
criticallimitations:⃝1 Robustnessvulnerabilitytosingleanomalies. ⃝2 Neglectsinstruction-response
variations. ⃝3 Ignorestop-Krankingimportance. ⃝4 Insensitivetohigh-rankchanges. ⃝5 Inadequate
handlingofedgecases. AdetailedanalysisoftheselimitationsisprovidedinAppendixB.
Wedefinetwometricstoquantifythemodel’sresponsivenesstoinstructions: StrictInstructionCom-
plianceRatio(SICR)andWeightedInstructionSensitivityEvaluation(WISE)metric. Assumingthat
intheoriginalmode,thecorequeryqhasnpositivedocuments,denotedasP={P 1 ,P 2 ,...,P n }.
Whenitcomestotheinstructedmodewherethecorequeryisdesignatesasinglegolddocument
Piout of P, demoting others to negatives. When In the reversely instructed mode, Pi becomes
denotePi’srankingsandS
negative. LetR ori ,R ins andR rev ori ,S ins andS rev itsrelevancescores
acrossoriginal,instructed,andreversedmodes.
StrictInstructionComplianceRatio TheSICRmetricintroducesastrictcriterionforevaluating
sensitivitytoinstructions. Ideally,foraretrievalresultthatstrictlyadherestotheinstruction,thegold
document’srankingandrelevancescoreintheinstructedmodeshouldbehigherthanintheoriginal
mode, denoted as (R < R & S > S ). Simultaneously, in the reversely instruction
|     |     | ins ori | ins | ori |     |     |     |
| --- | --- | ------- | --- | --- | --- | --- | --- |
mode,itsrankingandrelevancescoreshouldbelowerthanthoseintheoriginalmode,denotedas
(R < R & S > S ). Aquerythatstrictlysatisfiesthesecriteriaisassignedascoreof
| ori | rev | ori rev |     |     |     |     |     |
| --- | --- | ------- | --- | --- | --- | --- | --- |
1. Implementingrigorousscoringcriteriaensuresthatallchangesofrelevantdocumentsaretaken
intoaccount,therebyeffectivelyaddressingtheissueoflow-scoresensitivity(defect⃝1)andand
| incompleteevaluation(defect⃝2). |     |     | Theformulaforthiscriterionisasfollows: |     |     |     |     |
| ------------------------------- | --- | --- | -------------------------------------- | --- | --- | --- | --- |
(cid:26)
|       | 1,  | (R <R )and(S | >S  | )and(R | <R  | )and(S >S | ),  |
| ----- | --- | ------------ | --- | ------ | --- | --------- | --- |
| I(q)= |     | ins ori      | ins | ori    | ori | rev ori   | rev |
(1)
0, otherwise,
TheSICRscoreiscalculatedastheratioofthenumberofqueriesmeetingtheinstruction-following
criteriatothetotalnumberofqueriesinthetestset,representedbythefollowingformula:
(cid:80)J I(q )
j=1 j
|     |     |     | SICR= |     | ,   |     | (2) |
| --- | --- | --- | ----- | --- | --- | --- | --- |
|Q|
Where|Q|representsthetotalnumberofqueriesinthetestset. Thisformulacalculatethepercentage
ofretrievalsthatstrictlyadheretothespecifiedinstructionsrelativetothetotalresults.

PublishedasaconferencepaperatICLR2025
WeightedInstructionSensitivityEvaluation TheSICRmetricreflecttheproportionofmodel
resultsthatcomplywithinstructionsbutlacksadetailedquantificationofthedegreeofcompliance.
Onthisbasis,theWISEmetricrelaxestheevaluationcriteriabyfocusingonlyontherankingchanges,
regardstheresultsthatmeet(R ≤ R < R )2 asfollowingtheinstructions,andprovides
|     |     |     | ins | ori | rev |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
morelevelsofrewardsorpenaltiesfortheresults. Itcanbecalculatedusingthefollowingformula:
(cid:26)
|     |       |     | f reward (R | ori ,R ins | ),  | ifR ins    | ≤R ori <R | rev , |     |
| --- | ----- | --- | ----------- | ---------- | --- | ---------- | --------- | ----- | --- |
|     | F(q)= |     |             |            |     |            |           |       | (3) |
|     |       |     | f (R        | ,R         | ),  | otherwise, |           |       |     |
|     |       |     | penalty     | ori ins    |     |            |           |       |     |
When defining the reward component, the model is expected to comprehend and execute the in-
structions,effectivelyoptimizingtherankingsofthetopKresultsaccordingly. Thisimpliesthat
significantrankingchangeswithinthetopKresultsshouldbegivengreaterweighttoaddressdefects
⃝3 and⃝4, asthesechangesaremorelikelytobenoticedandutilizedbyusers. Itisessentialto
considerboththeabsoluterankingR andtherelativeranking(R −R ). Toachievethis,
|     |     |     |     | ins |     |     | ori | ins |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
weintroducedthe 1 term,generouslyrewardingsmallerR values. Simultaneously,through
|     |     | √   |     |     |     |     | ins |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
R ins
the(1− Rori−Rins)term,wegranthigherrewardstoresultsthatdemonstratesubstantialranking
K
improvements. Additionally,auniformvalueof0.01isassignedtoresultsbeyondtheTopK.More
extreme cases are considered (defects ⃝5): if a core query contains N positive documents in the
originalmodeandmeetstheconditionsR ≤ N andR = 1,itwillbegrantedarewardof1.
|     |     |     |     | ori |     | ins |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Thisisbasedonthepremisethat,foranidealretriever,theN positivedocumentswouldlikelyrank
atthetopintheoriginalmode. Accordingly,resultsthatrankhigherandexhibitmoresignificant
| changesshouldbeassignedgreaterweight. |     |     |     | Therewardformulaisdefinedas: |     |     |     |     |     |
| ------------------------------------- | --- | --- | --- | ---------------------------- | --- | --- | --- | --- | --- |

|     |     |  1, |     |     |     | ifR | ori ≤N | and R ins =1, |     |
| --- | --- | ----- | --- | --- | --- | --- | ------ | ------------- | --- |
√
|     |          |      | Ror i−Rins)· |     | 1     |        |            |     |     |
| --- | -------- | ---- | ------------ | --- | ----- | ------ | ---------- | --- | --- |
|     | f reward | = (1 | −            |     | √ ,   | i f R  | o r i ≤ K, |     | (4) |
|     |          |      | K            |     | R ins |        |            |     |     |
|     |          | 0  | .0 1,        |     |       |        | se,        |     |     |
|     |          |      |              |     |       | o th e | r w i      |     |     |
whereK =20signifiesthatourprimaryfocusisonthetop20retrievalresults. Someofthereward
valuesarevisualizedintheFigure5.
Forthepenaltycomponent,wereferencethedesignofp-MRR,emphasizingthemagnitudeofthe
rankingdropandapplystricterdemeritpointsforgolddocumentsthatexperiencealargerdecline
inranking. However,unlikep-MRR,ourR ins ,R ori ,andR rev yieldsixpossiblepermutations. To
accountfortheremainingfivecasesasidefrom(R ins ≤R ori <R rev ),weformulatedthefollowing
| penaltyformula: | 3   |     |     |     |     |     |     |     |     |
| --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

|     |     |         |       | −1,      | ifR | <R     | <R    | ,   |     |
| --- | --- | ------- | ----- | -------- | --- | ------ | ----- | --- | --- |
|     |     |         |     |          |     | rev    | ori   | ins |     |
|     |     |         |       | Ror i− R |     |        |       |     |     |
|     |     | f       | =     | ins,     | ifR | ori ≤R | ins , |     | (5) |
|     |     | penalty |       | R in s   |     |        |       |     |     |
|     |     |         | Rre | v− R     |     |        |       |     |     |
|     |     |         |       | ori,     | ifR | ≤R     | ,     |     |     |
|     |     |         |       | R        |     | rev    | ori   |     |     |
or i
Insummary,foratestsetwithJ queries,theoverallevaluationformulacanbeexpressedas:
|     |     |     |     |        | (cid:80)J | F(q ) |     |     |     |
| --- | --- | --- | --- | ------ | --------- | ----- | --- | --- | --- |
|     |     |     |     |        | j=1       | j     |     |     |     |
|     |     |     |     | WISE = |           | ,     |     |     | (6) |
J
3 EXPERIMENTS
Thissectionfirstintroducestheexperimentalsettingsin§3.1,followedbyadescriptionoftheoverall
retrievalresultsofdifferenttypesofsearchmodelsin§3.2. Lastly,itdiscussesmodels’performance
acrossindividualdimensionsin§3.3.
3.1 EXPERIMENTALSETUP
The goal of the benchmark is to determine how effectively retrieval models adjust their retrieval
behaviorinresponsetoinstructions. Tothoroughlyassesshowstate-of-the-artretrievalmodelsfollow
instructions,weevaluate15modelsacrossfourcategoriesofmodels:
Weselected15modelsrepresentingthefourmodelarchitectures:
2R maybeequalto1.
ori
3R
ori ≤ R ins covers two cases: (R ori ≤ R ins ≤ R rev ) and (R ori ≤ R rev ≤ R ins ). Similarly,
| R rev ≤R | ori coverstwocases:(R |     | rev ≤R | ori ≤R | ins )and(R | rev ≤R | ins ≤R | rev ). |     |
| -------- | --------------------- | --- | ------ | ------ | ---------- | ------ | ------ | ------ | --- |

PublishedasaconferencepaperatICLR2025
Table1: Performancecomparisonofdifferentretrievalmodelsaveragedoversixdimensions. The
lastthreecolumnsdisplaytherankingofthegolddocumentintheoriginalquery(R ),andthe
ori
relativerankchangeafterapplyingtheinstructedandreversedinstructions.
nDCG@10 Robustness@10 GoldDocumentRank
Model p-MRR↑ WISE↑ SICR↑
Ori Ins Rev Ori Ins Rev Rori Rins↓ Rrev↑
BM25 47.5 39.1 38.5 47.5 17.7 20.9 7.0 -12.0 0.0 18.4 18.0 18.3
DenseRetrieval
Bge-Large-v1.5 53.2 34.9 34.9 53.2 15.8 21.0 21.3 -29.5 1.0 20.4 25.0 24.9
E5-Large-v2 60.4 52.0 49.9 60.4 26.6 30.2 5.6 -23.3 0.8 14.7 12.8 13.9
Instructor-XL 62.6 38.4 39.3 62.6 17.5 23.4 30.4 -29.8 2.7 30.5 30.5 36.3
Mistral-ins-v0.2 19.4 25.5 29.2 19.4 8.5 12.7 -32.4 -49.2 0.0 236.0 153.0 153.1
GTE-Qwen2 43.6 43.1 48.5 43.6 18.7 26.5 -21.7 -39.0 0.1 104.3 75.3 71.3
E5-Mistral-ins 78.3 64.3 66.0 78.3 41.8 46.4 4.0 -16.3 2.8 6.6 5.4 5.6
GritLM 70.8 66.2 66.3 70.8 44.2 48.3 -4.3 -11.1 6.9 14.4 5.8 8.9
SFR-Embedding-2-R 70.7 62.2 60.1 70.7 40.7 43.2 4.8 -18.1 2.1 7.4 5.7 5.6
NV-Embed-v2 69.5 54.5 52.2 69.5 33.3 36.0 17.7 -13.5 2.8 8.1 8.7 9.3
Point-wiseReranking
Mistral-ins-v0.2 62.0 58.4 59.0 62.0 38.0 44.7 -2.3 4.1 8.1 6.5 4.7 8.8
Llama-3.1 74.8 66.8 65.4 74.8 46.1 49.2 11.5 14.4 19.3 5.4 3.7 8.2
FollowIR 72.4 66.3 65.5 72.4 46.2 50.0 4.1 13.4 12.5 5.5 3.8 7.6
List-wiseReranking(Fine-tuned)
Zephyr-beta 70.8 55.9 58.0 70.8 32.0 36.4 1.7 -3.2 8.7 6.4 6.1 7.0
RankVicuna-v1 65.4 55.2 55.2 65.4 31.2 35.7 2.0 -6.5 5.6 7.3 6.3 7.0
RankZephyr-v1 75.0 63.5 64.7 75.0 41.8 47.5 0.7 14.5 10.5 4.5 4.4 5.4
List-wiseReranking(InstructionalZero-shot)
Mistral-ins-v0.2 74.5 64.4 61.6 74.5 40.5 42.2 7.2 8.1 22.0 5.7 4.8 7.2
GPT-4o 83.8 74.2 74.2 83.8 53.0 58.0 15.0 33.5 32.1 2.6 1.7 4.3
• Sparseretrieval: 1model,BM25(Robertsonetal.,2009).
• Dense retrieval: 8 models, including BGE-large-v1.5 (Xiao et al., 2023), E5-large-v2
(Wangetal.,2022),Instructor-XL(Suetal.,2023),E5-Mistral(Wangetal.,2023),GritLM
(Muennighoffetal.,2024),NV-Embed-v2(Leeetal.,2024a),GTE-Qwen2(Lietal.,2023),
andSFR-Embedding-v2(Mengetal.,2024).
• Fine-tunedrankingmodels: 3models,includingFollowIR(Welleretal.,2024a),RankVi-
cuna (Pradeep et al., 2023a), RankZephyr (Pradeep et al., 2023b), where FollowIR is a
point-wisemodelandtheothertwoarelist-wise.
• Instruction-tunedgenerationmodelsusedforreranking: 3models,includingMistral-
7B-Instruct-v0.2(Jiangetal.,2023),Zephyr(Tunstalletal.,2023),andGPT-4o(Achiam
etal.,2023).
Fordenseretrievalmodels, wecomputethedotproductbetweenqueryanddocumentvectorsto
determineretrievalrankings. Forrerankingmodels,thetop100resultsfromE5-mistral(Wangetal.,
2023)arere-rankedbasedonthemodels’interpretationoftheinstruction. Forgenerallargelanguage
models,weusetwosettings: Inthepoint-wisesetting,boththequeryanddocumentareinputs,with
theoutputprobabilitiesofTrueorFalseusedassimilarityscores. Inthelist-wisesetting,following
(Pradeepetal.,2023b),alistofdocumentsisprovidedasaprompt(seeAppendixC),andthemodel
returnstherankeddocumentIDsinalist.
Based on Mistral-7B-Instruct-v0.2 (Jiang et al., 2023), we conduct a specialized experiment to
evaluateitszero-shotperformanceinthreeretrievalsettings: denseretrieval,point-wisereranking,
andlist-wisereranking. Asahighlycapableinstruction-tunedmodel,Mistralisexpectedtodemon-
strateinstruction-followingabilitiesinretrievaltaskswithoutfine-tuning. Thisexperimentexplores
Mistral’spotentialasazero-shotretrievalmodel,assessingwhetheritcannaturallygeneratestrong
embeddingsoractasaneffectivererankertoidentifyinstruction-relevantdocumentsfromthelist
ofcandidates. Forthemodeofdenseretrieval,weusemeanpoolingtoobtainthesentencelevel
representation.
WeincludeGPT-4oasastrongbaselineduetoitsdemonstratedinstruction-followingcapabilities
andtosetahighperformancebenchmarkforallmodels.

PublishedasaconferencepaperatICLR2025
3.2 RESULTOVERVIEW
Table1providesadetailedcomparisonofdifferentretrievalmodelsacrosssixdimensionsusing
nDCG@10,Robustness@10,p-MRR,WISE,andSICR.Thetablealsoincludestheaveragerankings
ofthegoldendocumentsintheinstructionmodeacrossthethreeretrievalmodels. Notably,almost
allmodelsachievedrelativelyhighnDCG,indicatingthatrelyingsolelyonnDCGisinsufficientto
capturetheimpactofinstructionsonrankingchanges. AlthoughRobustnesscanbeusedformodel
comparison,itisunabletoassesstheextentofperformancechangesbeforeandafterinstructions
because the relevant documents corresponding to the three retrieval modes differ. p-MRR can
partiallyreflectthemodel’sresponsivenesstodifferentinstructions;however,duetothelimitations
of this metric, the results are not expressed with sufficient accuracy. For instance, according to
p-MRRevaluations,theinstruction-followingperformanceofbge-large-v.5andInstructormodels
issignificantlybetterthanthatofGPT-4o. Meanwhile, theWISEandSICRscorescloselyalign
withtherankingchangesoftheGoldDocumentandcanclearlydistinguishtheinstruction-following
capabilitiesofthemodelsaswellastheperformancedifferencesbetweenthem. Theresultsreveal
distinctpatternsininstruction-followingperformanceacrossdifferentmodelcategories,whichcanbe
summarizedasfollows: list-wisererankingmodels>point-wisererankingmodels>denseretrieval
models>sparseretrievalmodels. Largermodelarchitecturestypicallyoutperformsmallermodelsin
bothWISEandSICR.
TheWISEandSICRscoresofthesparseretrievalmodelBM25indicatethatmodelsrelyingsolely
onlexicalmatching,withoutsensitivitytoinstruction-basedretrievalorcontext-awareinstructions,
struggle to interpret and act on complex instructions. BM25’s inability to adapt underscores the
limitationsoftraditionalsparseretrievalforinstruction-followingtasks.
Incontrast,denseretrievalmodelsshowgreatersensitivitytoinstructions,thoughtheirperformance
varies. Forinstance,BGE-Large-v1.5andInstructor-XLdemonstratesignificantperformancedegra-
dationunderinstructions,asreflectedintheirnegativeWISEscores. However,modelslikeGritLM,
E5-Mistral-insandNV-Embed-v2demonstrategreateradaptability. Notably,GritLMachievesthe
highestWISEandSICRscoreamongthedensemodels,indicatingthat,benefitingfromjointtraining
onbothencodingandgenerativetasks,GritLMisbetterequippedtohandlecomplexinstructions.
In contrast, models primarily trained on task-specific instructions, such as BGE-Large-v1.5 and
Instructor-XL,encounterdifficultieswhenaddressingabroaderrangeofinstructions.
Point-wisererankingmodelsgenerallyoutperformdenseretrievalmodels. Amongthem, Llama-
3.1achievesthehighestWISEandSICRscores. Althoughithasnotbeenspecificallyfine-tuned
forretrievaltasks,Llama-3.1benefitsfromitsextensiveunderstandingoflanguage,grantingita
certaindegreeofinstruction-followingcapabilities. FollowIRalsodemonstratescompetitiveness;by
fine-tuningwithcontent-awareinstructions,FollowIRachievescomparablescorestoLlama-3.1with
fewermodelparameters.
Amonglist-wisererankingmodels,GPT-4operformsthebest,achievingthehighestscoresinWISE
andSICRacrossallmodels,demonstratingitsexceptionalcapabilityinhandlingandadheringto
complexinstructions. Additionally,RankZephyrshowsdecentperformancebutremainscloserto
pointwise re-ranking models in terms of instruction following, possibly due to limitations in its
trainingdata. AlthoughMistral-ins-v0.2hasthesecond-highestSICRscoreafterGPT-4o,itsWISE
scoreisnotasremarkable,indicatingthatwhilethemodelcancomprehendinstructions,itstruggles
toeffectivelyelevatetherankingsofthecorrespondingdocuments.
3.3 PERFORMANCEANALYSISACROSSDIMENSIONS
The radar plots in Figure 3 offer a visual summary of how different models perform in these
dimensions,highlightingtheirstrengthsandweaknessesininstructionfollowing. Acrossallmodels,
bothretrievalandrerankingmodelsshowsignificantroomforimprovement. Particularly,certain
dimensions–suchasformatandaudience–consistentlypresentchallenges. Performanceonthese
remainssuboptimal,indicatingthatmodelsstrugglewithinstructionsrequiringspecificformatting
or audience adaptation. The difficulty likely arises from insufficient exposure to structured data
formatssuchas[StackOverflowPost],[CodeSnippet],or[OfficalManual],andalackofnuanced
understandingofdiverseaudiencecontextsduringtraining.

PublishedasaconferencepaperatICLR2025
GritLM NV-Embed-v2 SFR-Embedding-2 GPT-4o Llama-3.1 Mistral (list-wise)
Bm25 E5-Mistral E5-Large-v2 RankZephyr FollowIR Mistral (point-wise)
| Language | Length    |        | Language | Length    |        |
| -------- | --------- | ------ | -------- | --------- | ------ |
|          |           | 60     |          |           | 60     |
|          |           | 40     |          |           | 40     |
|          | 20        |        |          | 20        |        |
|          | 0         |        |          | 0         |        |
|          | -20       |        |          | -20       |        |
| Format   | -40       | Source | Format   | -40       | Source |
| Keyword  | Audience  |        | Keyword  | Audience  |        |
| (a)      | Retrieval |        | (b)      | Reranking |        |
Figure3: RadarplotscomparingtheWISEscoresofvariousmodelsacrossdifferentdimensions,
highlightingthestrengthsandweaknessesofeachmodelinhandlingdifferenttypesofinstructions.
Amongretrievalmodels,GritLMdemonstratesthestrongestinstruction-followingcapability,while
GPT-4consistentlyperformsthebestacrossalldimensionsinthererankingcategory.
Retrievalmodelsshownotablevariabilityinperformanceacrossdimensions. GritLMstandsout,
leading in overall instruction-following ability. Retrieval models generally perform well on the
language and keyword dimensions, but they struggle significantly on the format and audience
dimensions. Thisindicatesthatretrievalmodelshandletext-basedinstructionseffectivelybutstruggle
withstructuralandcontextualcues.
Comparedtoretrievalmodels,rerankingmodelsgenerallyperformbetteracrossalldimensions. This
improvementisparticularlyevidentinthekeyworddimension,largelybecausererankingmodels,
during inference, directly verify keyword presence within the context. GPT-4 stands out in the
language,source,andkeyworddimensions,consistentlyoutperformingothermodels. However,even
top-performingmodelslikeGPT-4ofacechallengeswithaudience-relatedinstructions. Despitethe
overallperformancegapintheSourceandAudiencedimensions,RankZephyrperformscomparably
to GPT-4o in the length, audience, and keyword dimensions, demonstrating the effectiveness of
fine-tuningforrerankingtasks.
4 ANALYSIS
p-MRRvs. WISE.Whilebothmetricsaimtomeasuremodels’instruction-followingabilitiesby
consideringrankchanges,p-MRRdoesnotconsistentlyreflectrealperformance.Manymodelsinthis
studyreceivedp-MRRscoresthatwereinconsistentwiththerankingtrendsofthegolddocuments;
for instance, GPT-4o scored 15.0, which was lower than both Instructor-XL and NV-Embed-v2.
Eventually,mostmodelsscoredevenlowerthanBM25. Thisdiscrepancyarisesbecausep-MRR
evaluatesonlyrelativerankingchanges(R −R ),disregardingabsoluterankingshifts(R ).
|     |     | ins | ori |     | ori |
| --- | --- | --- | --- | --- | --- |
Incontrast,theproposedWISEmetricstrictlyenforcesinstructionfollowingstandardsbyaccounting
forbothabsoluteandrelativerankingchanges. GPT-4oachievedthehighestWISEscore,asitwas
abletofurtherelevatetherankingsoftopgoldedocumentswheninstructionswereaddedandtolower
therankingsundersemanticallyinverseinstructions(R ori =3.51,R ins <R ori ,R ori <R rev ).This
makesWISEamorereliablemetricforevaluatinginstruction-followingcapabilities.
Dense retrieval model vs Reranking model. Reranking models(represented by red and gray
rowinTable2)significantlyoutperformmostdenseretrievalmodels(representedbygreenrowin
Table2)ininstruction-followingtasksduetotheirabilitytoevaluatedocumentsinrelationtoone
another, optimizing the final ranking based on contextual relevance and nuanced understanding.
By considering the entire list of retrieved documents, rerankers can effectively adjust the order
basedonthespecificneedsofthequery,capturingsubtledistinctionsthatdenseretrievalmodels
may overlook. This leads to more accurate rankings that align with user intent, especially in

PublishedasaconferencepaperatICLR2025
Table2: PerformancecomparisonofdifferentretrievalmodelsacrosssixdimensionsusingtheWISE
andSICRmetrics. Thedimensionsare: D1(Audience),D2(Keyword),D3(Format),D4(Language),
D5(Length),andD6(Source). Higherscoresindicatestrongerinstruction-followingcapabilities.
WISE SICR
Model
D1 D2 D3 D4 D5 D6 Avg. D1 D2 D3 D4 D5 D6 Avg.
BM25 -3.0 -42.1 -2.8 -7.2 -7.5 1.97 -12.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
DenseRetrieval
Bge-Large-v1.5 -16.8 -38.2 -42.1 -20.7 -28.6 -30.7 -29.5 0.5 0.0 0.3 2.0 1.0 2.3 1.0
E5-Large-v2 -15.6 -38.3 -15.5 -25.3 -21.6 -23.2 -23.3 1.4 0.7 0.0 0.5 0.7 1.3 0.8
Instructor-XL -27.7 -34.7 -30.5 -20.5 -35.5 -29.8 -29.8 5.7 2.1 0.3 4.0 0.3 4.0 2.7
Mistral-ins-v0.2 -35.8 -67.8 -29.7 -31.9 -66.6 -63.3 -49.2 0.0 0.0 0.0 0.0 0.0 0.0 0.0
GTE-Qwen2 -34.0 -36.5 -44.3 -18.0 -56.4 -44.6 -39.0 0.0 0.0 0.0 0.0 0.3 0.0 0.1
E5-Mistral-ins -7.3 -44.5 -19.9 0.1 -13.4 -13.0 -16.3 2.9 0.0 0.0 10.5 0.0 3.3 2.8
GritLM -3.4 6.8 -36.0 -6.7 -25.8 -1.5 -11.1 11.4 11.8 1.3 4.5 0.3 11.7 6.9
SFR-Embedding-2-R -7.8 -45.9 -13.0 -15.4 -13.5 -13.2 -18.1 2.9 1.0 2.0 1.0 1.0 4.7 2.1
NV-Embed-v2 -9.8 -27.7 -18.1 -7.3 -9.3 -8.7 -13.5 2.4 0.7 1.0 3.5 0.3 9.0 2.8
Point-wiseReranking
Mistral-ins-v0.2 -8.9 34.5 -9.3 21.4 -12.6 -0.3 4.1 1.4 28.6 2.7 6.5 4.7 4.7 8.1
Llama-3.1 -6.2 38.7 -9.5 29.0 -5.9 40.2 14.4 6.2 38.3 10.0 22.0 2.7 36.7 19.3
FollowIR -2.3 47.7 -2.3 20.9 -2.6 18.8 13.4 3.3 27.2 7.0 19.5 1.7 16.3 12.5
List-wiseReranking
Mistral-ins-v0.2 -6.3 46.0 -6.6 7.6 -1.9 10.0 8.1 10.5 59.2 9.7 23.0 8.0 21.7 22.0
Zephyr-beta -2.7 14.1 -13.9 -6.9 -5.7 -3.9 -3.2 1.0 27.5 8.0 10.5 2.0 3.0 8.7
RankVicuna-v1 -2.5 -8.5 -9.8 -11.8 -4.3 -2.2 -6.5 5.2 10.5 3.3 4.5 2.3 8.0 5.6
RankZephyr-v1 7.4 53.9 7.8 10.6 7.8 -0.3 14.5 4.3 42.5 1.0 5.5 4.3 5.3 10.5
GPT-4o 7.4 63.0 21.9 53.1 10.2 45.2 33.5 15.2 60.6 11.3 55.5 10.3 39.7 32.1
complexscenarioswheretherelationshipbetweendocumentsandthequeryiscrucialforeffective
instruction-following. Consequently, while dense retrieval models excel in efficiently retrieving
relevantdocuments,rerankingmodelsprovidetheprecisionnecessarytoenhancetheoverallranking
quality,resultinginsuperiorperformanceintasksrequiringsophisticatedlanguagecomprehension.
Point-wise reranking vs. List-wise reranking. Point-wise reranking evaluates each document
independently, predicting relevance without considering other documents. In contrast, list-wise
rerankingconsiderstheentireset,optimizingtheoverallrankingorderbyleveragingrelativerelation-
shipsbetweendocuments. AsshowninTable2,list-wiseranking(grayrow)generallyoutperforms
point-wiseranking(redrow)forinstruction-followingtasks,asitbettercapturesthebroaderquery
contextandtherelativeimportanceofdocuments. Thismakeslist-wisererankingmoreeffectivein
organizingdocumentstoalignwithcomplexinstructions,improvingrelevanceandcoherenceacross
differentqueries.
Zephyrvs. RankZephyr. RankZephyroutperformsZephyrinbothWISEandSICRbecauseofits
moresophisticatedtrainingprocess,betterrobustnesstoinitialdocumentorder,multiplereranking
passesthathelpcorrectrankingerrors. ComparedtoZephyr,RankZephyrlearnsfromRankGPT,
whichallowsittoadoptmoresophisticatedrankingstrategies. Besides,RankZephyrbenefitsfrom
multiplepasses,allowingittoadjusttherankingmoreeffectivelycomparedtoasingle-passstrategy
like Zephyr’s, which might not optimize the ranking as thoroughly. These factors combine to
ensurethatRankZephyrminimizespenaltiesforrankingimportantdocumentstoolow,leadingto
significantlybetterWISEandSICRscores.
Mistral(retrieval)vs.(point-wise)vs(list-wise).Mistral-ins-v0.2showspoorretrievalperformance,
highlighting its limitations in handling complex, instruction-driven ranking scenarios. Without
specifictrainingforretrieval,itfailstorankdocumentseffectivelyonnuancedinstructions. Onthe
otherhand,Mistral-ins-v0.2inpoint-wiserankingdemonstratesimprovedperformanceinbothWISE
andSICR,asitscoresindividualdocumentsindependently,allowingittobetteradheretoinstructions,
thoughitlacksthedepthtoconsiderrelationshipsbetweendocuments. However,Mistral-ins-v0.2
in list-wise ranking truly excels, as it optimizes the entire list and takes document interactions
intoaccount,enablingittohandlemoresophisticatedinstruction-followingtasks. Thisresultsin
significantly better WISE and SICR scores, making list-wise Mistral-ins-v0.2 the most effective
approachforinstruction-followingtaskswhererankingcoherenceiscritical.

PublishedasaconferencepaperatICLR2025
5 RELATED WORK
DenseRetrieval Thedevelopmentofdenseretrievalmodelshassignificantlyenhancedtheseman-
ticunderstandingandefficiencyofretrievalsystems. Existingdensemodelscanbecategorizedinto
twotypesbasedontheirarchitecture:BidirectionalEmbeddingModelsandDecoder-onlyEmbedding
Models. BidirectionalEmbeddingModelsaretypicallybaseonBERT(Devlin,2018)orT5(Raffel
etal.,2020)encoders,performinggeneralembeddingtasks. EarlymodelsthatbaseonBERTorT5
forefficienttextembeddingsincludeSentenceBERT(Reimers,2019),SimCSE(Gaoetal.,2021)
and Sentence T5 (Ni et al., 2021). To better accommodate the requirements of text embeddings,
researchershavepre-trainedtheseencodersusingcontrastivelearning(Izacardetal.,2021;Wang
etal.,2022).Furthermore,thesemodelsarefine-tunedusingvarioussuperviseddatasetstoenhance
theirperformanceinretrievaltasksorotherdownstreamapplications(Leeetal.,2024b;Li&Li,
2023). Compared to bidirectional embedding models, decoder-only embedding models initially
performrelativelypoorlyingeneralembeddingtasks,primarilyduetotheirlimitedcapacitytocom-
prehensivelycaptureandutilizecontextualinformation(Brown,2020). However,manyresearchers
havesoughttooptimizethesemodels’performancebyintroducingcontrastivelearningmethodsto
addresstheirdeficienciesinembeddingtasks(Neelakantanetal.,2022). Currently,researchershave
explorednotonlytheuseofsyntheticdata(Wangetal.,2023)butalsoahybridstrategycombining
realandsyntheticdata(Mengetal.,2024;BehnamGhaderetal.,2024),achievingsignificantsuccess
intextembeddingtasks. Collectively,theseadvancementsincontrastivepre-training,modelscaling,
andleveragingweaksupervisionandsyntheticdatasignificantlypropelthefieldofretrieval.
Instruction-FollowingforRetrieval Thenotionofrelevanceoftenvariesamongusers(Mizzaro,
1998). Consequently,queriesalonemaynotfullyaddressallusers’informationneeds(Ruthven&
Lalmas,2003),whileinstructionscanexpandtheseintentionsbeyondthescopeofthequeries.Recent
informationretrievalresearchhasrecognizedthisandtriedtotrainretrievalmodelsbycombining
instructions with queries to enhance their instruction-following capabilities. In general, existing
instruction-followingmodelscanbedividedintotwocategoriesbasedoninstructiondesignmethods:
task-awareandcontent-awareinstructionretrievalmodels. TARTfirstproposedageneralretrieval
systemwithtask-levelinstruction,settingspecificinstructionsfordifferentretrievaltaskstoquery
correspondingresults(Asaietal.,2023). Subsequently,Instructorexpandedthescopeofinstructions
so that text embeddings can not only retrieve but also classify and diagnose duplicate problems
(Su et al., 2023). However, these task-aware instructions are too general and lack the specificity
ofuserinstructionsinrealscenarios. Onthisbasis,otherresearchershavedevelopedcontent-level
instructions. InstructIRsetinstructionstoadapttoquery-textpairsbasedonuserbackground(such
as work, hobbies) (Oh et al., 2024). ExcluIR set exclusionary instructions based on the content
differencesbetweenqueryresults,accountingforusers’exclusionaryneedsinqueries(Zhangetal.,
2024). FollowIRsetinstructionstodistinguishqueryresultsbycombiningexclusionandinclusion
(Welleretal.,2024a). PIRfocusesontheabilityofretrieverstorecognizeandrespondtodifferent
perspectivesinqueries(?). However,realuserintentionsinvolvebothinternal(content,audience,
language)andexternal(format,length)answerattributes. MAIRproposedalarge-scaleinstruction
retrievalbenchmarkcovering126differentinformationretrievaltasks,butlacksanexplicitevaluation
ofthemodel’sinstructionfollowingability(?).
6 CONCLUSION
Despite leveraging LMs as the backbone for training retrieval models, most existing IR models
cannottrulyunderstandtheinstructionsinquery. Further,traditionalscoreindicators(e.g.,nDCG)
cannotreflectwhetherthemodelhastheabilitytofollowinstructionsandmostexistingdatasetwith
instructionsaredesignedwithaonlysingledimension,soweproposeInfoSearchandtwonovel
metrics(WISE,SICR).Thechoiceofdimensionsinourdatasettakesintoaccounttheinstructions
thatusersmaygiveinactualscenarios. Additionally,ourmetricsconsiderthecombinedperformance
of the model in three modes(Original mode, Instructed mode, Reversely instructed mode), with
increasingdifficultyacrossmodesaseachintroducesmorecomplexchallenges. Wehopethiswork
helpsthefutureinstruction-followingretrievaltask.

PublishedasaconferencepaperatICLR2025
ACKNOWLEDGEMENT
WethankEITandIDTHighPerformanceComputingCenterforprovidingcomputationalresources
forthisproject. Thisworkissupportedby2035KeyResearchandDevelopmentProgramofNingbo
CityunderGrantNo.2024Z127.
REFERENCES
JoshAchiam,StevenAdler,SandhiniAgarwal,LamaAhmad,IlgeAkkaya,FlorenciaLeoniAleman,
DiogoAlmeida,JankoAltenschmidt,SamAltman,ShyamalAnadkat,etal. Gpt-4technicalreport.
arXivpreprintarXiv:2303.08774,2023.
AkariAsai,TimoSchick,PatrickLewis,XilunChen,GautierIzacard,SebastianRiedel,Hannaneh
Hajishirzi,andWen-tauYih. Task-awareretrievalwithinstructions. InFindingsoftheAssociation
forComputationalLinguistics: ACL2023,pp.3650–3675,2023.
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge,
YuHan,FeiHuang,etal. Qwentechnicalreport. arXivpreprintarXiv:2309.16609,2023.
PayalBajaj,DanielCampos,NickCraswell,LiDeng,JianfengGao,XiaodongLiu,RanganMa-
jumder,AndrewMcNamara,BhaskarMitra,TriNguyen,etal. Msmarco: Ahumangenerated
machinereadingcomprehensiondataset. arXivpreprintarXiv:1611.09268,2016.
ParishadBehnamGhader,VaibhavAdlakha,MariusMosbach,DzmitryBahdanau,NicolasChapados,
andSivaReddy. Llm2vec: Largelanguagemodelsaresecretlypowerfultextencoders. arXiv
preprintarXiv:2404.05961,2024.
SidBlack,StellaBiderman,EricHallahan,QuentinAnthony,LeoGao,LaurenceGolding,HoraceHe,
ConnorLeahy,KyleMcDonell,JasonPhang,etal. Gpt-neox-20b: Anopen-sourceautoregressive
languagemodel. arXivpreprintarXiv:2204.06745,2022.
TomBBrown. Languagemodelsarefew-shotlearners. arXivpreprintarXiv:2005.14165,2020.
JacobDevlin. Bert:Pre-trainingofdeepbidirectionaltransformersforlanguageunderstanding. arXiv
preprintarXiv:1810.04805,2018.
Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence
embeddings. arXivpreprintarXiv:2104.08821,2021.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand
Joulin,andEdouardGrave. Unsuperviseddenseinformationretrievalwithcontrastivelearning.
arXivpreprintarXiv:2112.09118,2021.
AlbertQJiang,AlexandreSablayrolles,ArthurMensch,ChrisBamford,DevendraSinghChaplot,
DiegodelasCasas,FlorianBressand,GiannaLengyel,GuillaumeLample,LucileSaulnier,etal.
Mistral7b. arXivpreprintarXiv:2310.06825,2023.
ChankyuLee,RajarshiRoy,MengyaoXu,JonathanRaiman,MohammadShoeybi,BryanCatanzaro,
andWeiPing. Nv-embed: Improvedtechniquesfortrainingllmsasgeneralistembeddingmodels.
arXivpreprintarXiv:2405.17428,2024a.
Sean Lee, Aamir Shakir, Darius Koenig, and Julius Lipp. Open source strikes bread -
new fluffy embeddings model, 2024b. URL https://www.mixedbread.ai/blog/
mxbai-embed-large-v1.
XianmingLiandJingLi. Angle-optimizedtextembeddings. arXivpreprintarXiv:2309.12871,2023.
ZehanLi,XinZhang,YanzhaoZhang,DingkunLong,PengjunXie,andMeishanZhang. Towards
generaltextembeddingswithmulti-stagecontrastivelearning. arXivpreprintarXiv:2308.03281,
2023.
RenzeLou, KaiZhang, JianXie, YuxuanSun, JaniceAhn, HanziXu, YuSu, andWenpengYin.
Muffin: Curatingmulti-facetedinstructionsforimprovinginstructionfollowing. InTheTwelfth
InternationalConferenceonLearningRepresentations,2023.

PublishedasaconferencepaperatICLR2025
Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. Sfr-
embedding-2: Advanced text embedding with multi-stage training, 2024. URL https://
huggingface.co/Salesforce/SFR-Embedding-2_R.
StefanoMizzaro. Howmanyrelevancesininformationretrieval? Interactingwithcomputers,10(3):
303–320,1998.
NiklasMuennighoff,NouamaneTazi,Lo¨ıcMagne,andNilsReimers. Mteb:Massivetextembedding
benchmark. arXivpreprintarXiv:2210.07316,2022.
NiklasMuennighoff,HongjinSu,LiangWang,NanYang,FuruWei,TaoYu,AmanpreetSingh,and
DouweKiela. Generativerepresentationalinstructiontuning. arXivpreprintarXiv:2402.09906,
2024.
ArvindNeelakantan,TaoXu,RaulPuri,AlecRadford,JesseMichaelHan,JerryTworek,Qiming
Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. Text and code embeddings by
contrastivepre-training. arXivpreprintarXiv:2201.10005,2022.
JianmoNi,GustavoHernandezAbrego,NoahConstant,JiMa,KeithBHall,DanielCer,andYinfei
Yang. Sentence-t5:Scalablesentenceencodersfrompre-trainedtext-to-textmodels. arXivpreprint
arXiv:2108.08877,2021.
HanseokOh,HyunjiLee,SeonghyeonYe,HaebinShin,HansolJang,ChangwookJun,andMinjoon
Seo. Instructir: Abenchmarkforinstructionfollowingofinformationretrievalmodels. arXiv
preprintarXiv:2402.14334,2024.
LetianPeng,YuweiZhang,ZilongWang,JayanthSrinivasa,GaowenLiu,ZihanWang,andJingbo
Shang. Answerisallyouneed: Instruction-followingtextembeddingviaansweringthequestion.
arXivpreprintarXiv:2402.09642,2024.
RonakPradeep,SahelSharifymoghaddam,andJimmyLin.Rankvicuna:Zero-shotlistwisedocument
rerankingwithopen-sourcelargelanguagemodels. arXivpreprintarXiv:2309.15088,2023a.
Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. Rankzephyr: Effective and robust
zero-shotlistwisererankingisabreeze! arXivpreprintarXiv:2312.02724,2023b.
ColinRaffel,NoamShazeer,AdamRoberts,KatherineLee,SharanNarang,MichaelMatena,Yanqi
Zhou,WeiLi,andPeterJLiu. Exploringthelimitsoftransferlearningwithaunifiedtext-to-text
transformer. Journalofmachinelearningresearch,21(140):1–67,2020.
N Reimers. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint
arXiv:1908.10084,2019.
StephenRobertson,HugoZaragoza,etal. Theprobabilisticrelevanceframework: Bm25andbeyond.
FoundationsandTrends®inInformationRetrieval,3(4):333–389,2009.
IanRuthvenandMouniaLalmas. Asurveyontheuseofrelevancefeedbackforinformationaccess
systems. TheKnowledgeEngineeringReview,18(2):95–145,2003.
Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih,
NoahASmith,LukeZettlemoyer,andTaoYu. Oneembedder,anytask: Instruction-finetuned
textembeddings. InFindingsoftheAssociationforComputationalLinguistics: ACL2023,pp.
1102–1121,2023.
RobertSTaylor. Theprocessofaskingquestions. Americandocumentation,13(4):391–396,1962.
GeminiTeam,RohanAnil,SebastianBorgeaud,YonghuiWu,Jean-BaptisteAlayrac,JiahuiYu,Radu
Soricut,JohanSchalkwyk,AndrewMDai,AnjaHauth,etal. Gemini: afamilyofhighlycapable
multimodalmodels. arXivpreprintarXiv:2312.11805,2023.
HugoTouvron,ThibautLavril,GautierIzacard,XavierMartinet,Marie-AnneLachaux,Timothe´e
Lacroix, BaptisteRozie`re, NamanGoyal,EricHambro, FaisalAzhar,etal. Llama: Openand
efficientfoundationlanguagemodels. arXivpreprintarXiv:2302.13971,2023.

PublishedasaconferencepaperatICLR2025
LewisTunstall,EdwardBeeching,NathanLambert,NazneenRajani,KashifRasul,YounesBelkada,
ShengyiHuang,LeandrovonWerra,Cle´mentineFourrier,NathanHabib,etal. Zephyr: Direct
distillationoflmalignment. arXivpreprintarXiv:2310.16944,2023.
LiangWang,NanYang,XiaolongHuang,BinxingJiao,LinjunYang,DaxinJiang,RanganMajumder,
andFuruWei. Textembeddingsbyweakly-supervisedcontrastivepre-training. arXivpreprint
arXiv:2212.03533,2022.
LiangWang,NanYang,XiaolongHuang,LinjunYang,RanganMajumder,andFuruWei. Improving
textembeddingswithlargelanguagemodels. arXivpreprintarXiv:2401.00368,2023.
OrionWeller,BenjaminChang,SeanMacAvaney,KyleLo,ArmanCohan,BenjaminVanDurme,
DawnLawrie,andLucaSoldaini. Followir: Evaluatingandteachinginformationretrievalmodels
tofollowinstructions. arXivpreprintarXiv:2403.15246,2024a.
Orion Weller, Benjamin Van Durme, Dawn Lawrie, Ashwin Paranjape, Yuhao Zhang, and Jack
Hessel. Promptriever: Instruction-trainedretrieverscanbepromptedlikelanguagemodels. arXiv
preprintarXiv:2409.11136,2024b.
ShitaoXiao,ZhengLiu,PeitianZhang,NiklasMuennighoff,DefuLian,andJian-YunNie. C-pack:
Packagedresourcestoadvancegeneralchineseembedding. arXivpreprintarXiv:2309.07597,
2023.
WenhanXiong,XiangLorraineLi,SriniIyer,JingfeiDu,PatrickLewis,WilliamYangWang,Yashar
Mehdad,Wen-tauYih,SebastianRiedel,DouweKiela,etal. Answeringcomplexopen-domain
questionswithmulti-hopdenseretrieval. arXivpreprintarXiv:2009.12756,2020.
Wenhao Zhang, Mengqi Zhang, Shiguang Wu, Jiahuan Pei, Zhaochun Ren, Maarten de Rijke,
ZhuminChen,andPengjieRen. Excluir: Exclusionaryneuralinformationretrieval. arXivpreprint
arXiv:2404.17288,2024.

PublishedasaconferencepaperatICLR2025
A MORE DETAILS OF INFOSEARCH
A.1 CONDITIONDETERMINATION
Thequeryforeachdimensioncanbediversifiedandexpandedthroughvariousconditions,allowing
thesamequerytocorrespondtodifferentrelevantdocumentsunderdifferentconditions. Except
fortheKeyworddimension,theotherfivedimensionshavefixedconditions. Theconditionofthe
Keyword dimension is the keyword in the document that is relevant to the query. Therefore, the
conditionoftheKeyworddimensionneedstobedeterminedafterfilteringouttheQuery-Document
(Q-D)pairs. Toachievethis,boththequeryanditscorrespondingdocumentwereinputintoGPT-4,
generatingauniqueconditionforeachQ-Dpair. However,GPT-4occasionallyselectedirrelevant
words,suchas“and”or“what”,thatdidnotalignwiththeuserscenariowhengeneratingkeyword
conditions. To address this, we meticulously crafted prompt templates (Table 3) for condition
extraction, ensuring that the conditions were both unique and representative of each document,
accuratelyreflectingthedocument’scoretheme.
Table3: Atemplatethatgeneratesthespecificconditionsrequiredforthekeyworddimension
PromptforConditionGeneration
###TASK###
• Your task is to generate a condition that refines a given query in relation to a provided
document. Theconditionshouldbe:
1. Relevant to the document’s core topic – It must align with the central theme or key
contentofthedocument.
2. Ameaningfulconstraintonthequery–Itshouldintroduceaspecificaspect,subtopic,or
perspectivethatnaturallyextendstheoriginalquerywhilestillbeingdirectlysupported
bythedocument.
3. Notagenericorarbitraryrestriction–Theconditionmustbelogicallyderivedfromthe
document’scontentandshouldnotbeatrivialoroverlybroadconstraint.
###INPUT###
• Youwillreceiveaqueryandadocumentasinput:
– Query: {query}
– Document: {document}
###FORMATTING###
• Condition: <theconditionyourgenerated>
A.2 DATACOLLECTION
Toensurethequeriesrealisticandreducethehumancost,weconsciouslyintegrateconditionswhen
filteringQ-Apairsfromexistingdatasetsorwebpages. Forinstance,intheFormatdimension,dueto
thelackofavailablemulti-formatQ&Adatasets,weselectivelyextractQ-DpairsfromStackOverflow
posts. Forpostscontainingcodeanddetailedofficialdocumentationresponses,weusetheirtitlesas
queriesandthecompleteresponsesasdocumentsunderthe[StackOverflow]condition. Thepure
codesnippetswithintheanswersandreferencestoofficialdocumentationareseparatelyextracted
andusedasdocumentsunderthe[CodeSnippet]and[Manual]conditions, respectively. Table4
showsthesourceofdatasetsusedtocollectquery-documentpairsforeachdimension.
A.3 INSTRUCTIONGENERATION
Thegenerationofaccurate,concise,andnaturalinstructionsiscrucial. Whensearching,userstend
toexpresstheirintentionsusingsimple,naurallanguage,sothegeneratedinstructionsmustremain

PublishedasaconferencepaperatICLR2025
Table4: Structureandsourceofthedataset
| Dimension | SourceData                            | ConditionValue    |
| --------- | ------------------------------------- | ----------------- |
| Audience  | BioASQ,scifact(Muennighoffetal.,2022) | [Layman],[Expert] |
| Keyword   | MSMARCO(Bajajetal.,2016)              | [keyword]         |
Format Stackoverflow,variousofficedoc [StackoverflowPost],[CodeSnippet],[OfficialManual]
| Language | publichealth-qa | [Chinese],[English] |
| -------- | --------------- | ------------------- |
Length medicalqa(Muennighoffetal.,2022),googlesearch [Sentence],[Paragraph],[Article]
Source CNN-english-news,googlesearch [Blog],[ForumPost],[NewsArticle]
briefandclear,closelymirroringconversationalstyle. Toachievethis,weemployedwordssuch
as “smooth”, “natural”, and “realistic” in the prompts (see Table 5) to guide GPT-4 in crafting
instructions that emphasize not only semantic accuracy but also the emulation of authentic user
expressions. Furthermore,atwo-sentencestructurefortheinstructions,firstrephrasingthequery
andthenappendingspecificconstraints. Thisstructureeffectivelyseparatesthecorequeryfrom
the conditions, enhancing the diversity of generated instructions. For example, “What are the
mosteffectiveexercisesforlosingweight? Pleasefinddiscussionsfromforumpostsonly.” This
two-sentencestructureensureslogicalclarityandsemanticcoherence.
Table5: AprompttemplateforGeneratingInstruction
PromptforInstructionGeneration
###TASK###
• Youaretaskedwithgeneratinganaturalquerywithaninstructionbasedonthequeryandthe
conditionprovidedbytheuser. Youwillbeprovidedwithaqueryandaconditionandyou
needto:
1. Rephrase the core query as the first sentence, making it sound like a natural human
querywithoutchangingitsmeaning.
| 2.  | Createasecondsentencethatspecifiesthesearchrestriction. |     |
| --- | ------------------------------------------------------- | --- |
3. Ensureeachsentenceissmooth,concise,reasonable,natural,andrealistic,mimickinga
realhumantone.
###INPUT###
| • CoreQuery: | {corequery} |     |
| ------------ | ----------- | --- |
| • Condition: | {condition} |     |
###FORMATTING###
| • Corequery: | <thecorequeryIgiveyou>                                  |     |
| ------------ | ------------------------------------------------------- | --- |
| Condition:   | <theconditionIgiveyou>                                  |     |
| Query        | with Instruction: <thequerywithinstructionyougenerated> |     |
A.4 DOCUMENTREWRITING
Whenthequeryandrelevantdocumentsfailtomeettheinstructionrequirements, weuseGPT-4
to rewrite the existing documents to generate relevant documents. The documents that need to
be modified are mainly concentrated in the source, length and audience dimensions, so we set
specificpromptsforthesedimensionsrespectively(seeTable6). Inthisprocess,weexperimented
withdirectlygeneratingcondition-satisfyingdocumentsfromthequery,buttheseoftenexhibited
redundantexpressionsandinconsistentformatting. Therefore,weadjustedtheexistingdocuments,
ensuringtheymeettheinstructionrequirementswhilepreservingnaturalnessandauthenticityinthe
language.

PublishedasaconferencepaperatICLR2025
| Table6: | Prompttemplatesfordocumentrewriting |     |     |     |
| ------- | ----------------------------------- | --- | --- | --- |
Dimension PromptTemplate
Source ###TASK###
| • Foracorequery,Ineeddocumentsfromablog,forumpost,ornewsarticle. |     |     |     | I   |
| ---------------------------------------------------------------- | --- | --- | --- | --- |
willprovideyouwithacorequery,thecorrespondingdocumentfromanews
article. Yourtaskistorewritethedocumentasblogandforumpostcontent.
###CAUTION###
1. Fortheblogyougenerated,youcannotusethecorequeryasblogtitledirectly.
| Youneedtorephraseit,butdonotchangethesemanticsofthisquery. |     |     |           | Besides, |
| ---------------------------------------------------------- | --- | --- | --------- | -------- |
| youneedtogivevariousinformationinthelineunderthetitle,     |     |     | suchasthe |          |
author,whenitwaspublished,theword“Blog”,andthesectionitbelongsto.
Alltheaboveinformationmustberandom.
| 2. Foraforumpost,itmustbeaformofdiscussionamongmultipleusers. |     |     |     | The |
| ------------------------------------------------------------- | --- | --- | --- | --- |
usernamesneedtoberandomratherthanuse“use1”,“use2”etc.
###INPUT###
| • CoreQuery: | {corequery} |     |     |     |
| ------------ | ----------- | --- | --- | --- |
| • Document:  | {document}  |     |     |     |
###FORMATTING###
• Youroutputshouldbeinthefollowingformat:
• Blog: <theblogyougenerated>
Forum: <theforumpostyougenerated>
Audience ###TASK###
| • Iwillprovideyouwithacorequeryanditscorrespondingdocument. |                   |                  |                    | The  |
| ----------------------------------------------------------- | ----------------- | ---------------- | ------------------ | ---- |
| target audience                                             | for this document | is experts. Your | task is to Rewrite | this |
documenttomakeiteasilyunderstandableforlaymen.
###CAUTION###
1. Keepthesemanticsofthedocumentintact.
2. Donotuseanytechnicaljargonintherewrittendocumentforlayman.
###INPUT###
• Query: {query}
| • Documentforexpert: | {expert} |     |     |     |
| -------------------- | -------- | --- | --- | --- |
###FORMATTING###
<thequeryIgiveyou>
• Query:
Layman: <therewrittendocumentforlaymanyougenerated>
Length ###TASK###
| • Iwillprovideyouwithacorequeryanditscorrespondingparagraph. |     |     |     | Your |
| ------------------------------------------------------------ | --- | --- | --- | ---- |
taskistorewritethisparagraphintoasinglesentenceandanarticle.
###CAUTION###
1. Ensurethatboththesentenceandarticleretaintheoriginalmeaningofthe
paragraph.
###INPUT###
| • CoreQuery:             | {corequery} |     |     |     |
| ------------------------ | ----------- | --- | --- | --- |
| • Paragraph: {paragraph} |             |     |     |     |
###FORMATTING###
• Youroutputshouldbeinthefollowingformat:
| • Sentence: <therewrittensinglesentencethatanswersthequery> |     |     |     |     |
| ----------------------------------------------------------- | --- | --- | --- | --- |
Article: <themulti-paragraphrewrittendocumentthatanswersthequery>

PublishedasaconferencepaperatICLR2025
A.5 INSTRUCTIONREVERSAL
Inrealretrieval,weobservedthatresultsalreadyrankedhighlytendtoremainatthetopevenafter
instructionsareapplied. Thismakesitdifficulttodeterminewhethertheimprovementinranking
isduetothemodel’sunderstandingoftheinstructionsorsimplyaresultofdetailedkeywordand
semanticmatching. Toaddressthis,wevalidatethemodel’scomprehensionoftheinstructionsby
reversing the semantic meaning of the instructions. For example, “Please answer in Chinese” is
reversedto“PleasedonotanswerinChinese.”
Table7: ApromptTemplateforInstructionReversing
PromptforInstructionReversing
###TASK###
• Yourexpertiseliesininterpretingandtransformingdirectinstructionsintotheiroppositeor
negativeformswhilemaintainingclarityandcoherenceinthetransformedinstructions. Your
taskistoreversetheinstructionIgiveyou.
###CAUTION###
• Whilereversingtheinstruction,ensurethatthenewinstructionconveystheoppositemeaning
accurately. Please keep in mind that the transformation should remain clear and easy to
understand,avoidinganyambiguity.
###INPUT###
• Instruction: {instruction}
###FORMATTING###
• ReverseInstruction: <theinstructionyourreverse>
A.6 HARDNEGATIVEGENERATIONS
Whilepositivedocumentsforthesamequeryundervaryingconditionsmayactasnegativeexamples
foroneanother(instructionnegatives),westillneedtopreventthemodelfromrelyingsolelyon
prominentfeaturesforsimpleretrieval,therebyneglectingthesubtlerelationshipsbetweenthequery
andthedocuments. Toaddressthis,weuseGPT-4togeneratedocumentsthatappeartoberelated
to the query topic on the surface but cannot actually answer the query, serving as hard negative
documents(querynegatives).

PublishedasaconferencepaperatICLR2025
| Table8: | Aprompttemplatesforgeneratinghardnegative |     |     |     |
| ------- | ----------------------------------------- | --- | --- | --- |
PromptforHardNegativeGeneration
###TASK###
• Youaretaskedwithgeneratingahardnegativedocumentbasedonagivenquery. Ahard
negative document should appear superficially relevant to the query but contain critical
| inaccuracies,misleadingdetails,orsubtlecontradictions. |     |     | Followthesesteps: |     |
| ------------------------------------------------------ | --- | --- | ----------------- | --- |
1. Understand the core intent of the query and identify key entities, relationships, or
requirements.
2. Generate a document that incorporates some keywords from the query but does not
provide a direct or indirect answer to the query. The document should maintain a
plausiblestructureandstayonarelatedtopicwhileensuringthatnoinformationwithin
itcanbeusedtoinferorconstructacorrectresponsetothequery.
3. Ensure the document is coherent, natural, and realistic, mimicking a genuine but
incorrectresponse.
###INPUT###
• CoreQuery: {corequery}
###FORMATTING###
• Corequery: <thecorequeryIgiveyou>
| Hardnegativedocument: | <generateddocument> |     |     |     |
| --------------------- | ------------------- | --- | --- | --- |
A.7 MANUALREVIEW
Wefilteredoutanomalousdocumentsfromtheoutputsof12retrievalmodels,selectingthosethat
failedtorankwithinthetop50insixormoremodelsorhadarelevancescorebelow0.5forthequery,
followedbymanualscreening. ThisprocessaimedtoeliminatemislabeledQ-Dpairsselectedfrom
otherdatasetsordocumentsinaccuratelyretrievedthroughmanualsearch. Forthesemismatched
documents,weproceedwithmanualreplacement. Aftermultipleroundsofscreeningtoensurethe
qualityoftheInfoSearchdataset,thestatisticalresultsaresummarizedinTable9.
Table9: InfoSearchdatasetstatistics. |Q|, |I|and|R|representthewordlengthsoftheoriginal
query,instructedqueryandreverselyinstructedqueryrespectively.
| Dimension | #Q Avg|Q| | #I Avg|I| | #R Avg|R| | #D   |
| --------- | --------- | --------- | --------- | ---- |
| Audience  | 100 9.02  | 210 20.46 | 210 15.91 | 840  |
| Keyword   | 100 6.30  | 288 17.90 | 288 18.92 | 1152 |
| Format    | 100 9.16  | 300 16.65 | 300 19.31 | 1200 |
| Language  | 100 8.75  | 200 14.09 | 200 15.74 | 800  |
| Length    | 100 8.52  | 300 15.94 | 300 16.26 | 1200 |
| Source    | 100 7.38  | 300 18.19 | 300 15.58 | 1200 |
| Total     | 600       | 1598      | 1598      | 6392 |

PublishedasaconferencepaperatICLR2025
Tomakethedatamoreintuitive,Table10toTable15providespecificexamplesfromeachdimension
intheInfoSearchdataset.
|     | Table10: AnexampleinAudiencedimension |     |     |     |
| --- | ------------------------------------- | --- | --- | --- |
CoreQuery HowtoPreventHeartDisease
Instructed1 Exploreeffectivestrategiesforpreventingheartdisease. Pleaseexplaininterms
thatareeasyforthegeneralpublictounderstand.
Instructed2 Investigatethelatestpreventivemeasuresagainstheartdisease. Makeadetailed
discussionsuitableforaprofessionalaudience.
Reversed1 HowtoPreventHeartDisease. I’mlookingforaresponsethatismoretechnical
thanlayman.
Reversed2 HowtoPreventHeartDisease. Pleasekeepyouranswersimpleandclear.
Document1 Topreventheartdisease,considerthefollowingstrategies:
| AdoptaVeganDiet:                                            | Vegandiets,particularlythoserichinsoyandotherplant- |                 |                       |                 |
| ----------------------------------------------------------- | --------------------------------------------------- | --------------- | --------------------- | --------------- |
| basedproteins,canreducetheriskofcardiovasculardisease.      |                                                     |                 | Theseproteinsare      |                 |
| highinnon-essentialaminoacids,whichpromoteglucagonactivity. |                                                     |                 |                       | Glucagon        |
| helps regulate                                              | lipid levels                                        | and cholesterol | synthesis, leading to | healthier heart |
conditions.
| IncreaseGlucagonActivity: |     | ... |     |     |
| ------------------------- | --- | --- | --- | --- |
Document2 ... Veganproteinsmayreduceriskofcancer,obesity,andcardiovasculardisease
| by promoting | increased glucagon | activity. | ... glucagon promotes | (and insulin |
| ------------ | ------------------ | --------- | --------------------- | ------------ |
inhibits)cAMP-dependentmechanismsthatdown-regulatelipogenicenzymesand
cholesterolsynthesis,whileup-regulatinghepaticLDLreceptorsandproduction
| oftheIGF-IantagonistIGFBP-1. |     | Theinsulin-sensitizingpropertiesofmanyvegan |     |     |
| ---------------------------- | --- | ------------------------------------------- | --- | --- |
diets–highinfiber,lowinsaturatedfat...

PublishedasaconferencepaperatICLR2025
Table11: AnexampleinKeyworddimension
CoreQuery Whathelpsforacne?
Instructed1 Whattreatmentsareeffectiveforacne? Ensureyouranswerincludesinformation
specificallyabout“progesterone”.
Instructed2 Canyoutellmewhathelpsreduceacnesymptoms? Focusontheeffectsof“mint”
inyourresponse.
Instructed3 Whatnaturalremediesarebeneficialformanagingacne? Pleaseincludedetails
about“Chamomile”.
Reversed1 Whathelpsforacne? Canyouprovidearesponsethatdoesnotinvolvetheterm
“progesterone”?
Reversed2 Whathelpsforacne? Canyougivemeareplythatdoesnotentailtheuseofthe
term“mint”?
Reversed3 Whathelpsforacne?Canyouprovidearesponseavoidingtheterm“Chamomile”?
Document1 Progesterone helps with acne that occurs in the late 30’s and early 40’s. Also,
iftheacnevarieswiththeperiod,eliminationofxenoestrogens(environmental
estrogens)andphytoestrogensandtakingprogesteronecreamhelpswiththistype
ofacneaswell.
Document2 Acne home remedy: Mint. Mint can help remove pore-clogging oil. To help
clearacnebeforeitbegins,mix2tablespoonsoffinelychoppedfreshmintwith
twotablespoonseachofplainyogurtandoatmeal(useablendertopulverizethe
oatmealtopowder). Leavetheconcoctiononyourfacefor10minutes,thenrinse
offwithwater.
Document3 Acnehomeremedy: Chamomile. Chamomilehelpsdecreaseinflammationfrom
acne. Inablenderorcoffeegrinder,combinethecontentsofachamomileteabag
withenoughwatertoformapaste,andapplythattoacne. Alternately,steeptwo
chamomileteabagswith1cupboiledwaterfor15minutes.

PublishedasaconferencepaperatICLR2025
Table12: AnexampleinFormatdimension
CoreQuery HowcanIaccessenvironmentvariablesinPython?
Instructed1 HowcanIaccessenvironmentvariablesinPython? LimitthesearchtoStackover-
flowposts.
Instructed2 HowcanIaccessenvironmentvariablesinPython? Ineedcodesnippetstosolve
theproblem.
Instructed3 HowcanIaccessenvironmentvariablesinPython?Onlyconsiderofficialmanuals.
Reversed1 HowcanIaccessenvironmentvariablesinPython? Providemewithananswer
thatisnotaStackoverflowpost..
Reversed2 HowcanIaccessenvironmentvariablesinPython? Couldyoudeliveraresponse
thatisn’tintheformofacodesnippet?
Reversed3 HowcanIaccessenvironmentvariablesinPython? I’mseekingareplythatisn’t
anofficialmanual.
Document1 Environmentvariablesareaccessedthrough[‘os.environ‘]
“‘python
importos
print(os.environ[’HOME’])
“‘
Toseealistofallenvironmentvariables:
“‘python
print(os.environ)
“‘
Ifakeyisnotpresent,attemptingtoaccessitwillraisea‘KeyError‘. Toavoidthis:
“‘python
#Returns‘None‘ifthekeydoesn’texist
print(os.environ.get(’KEY THAT MIGHT EXIST’))
“‘
Document2 “‘python
importos
print(os.environ[’HOME’])
“‘
Document3 os.**environ**
A[mapping]objectwherekeysandvaluesarestringsthatrepresenttheprocess
environment. For example, ‘environ[’HOME’]‘ is the pathname of your home
directory(onsomeplatforms),andisequivalentto‘getenv(“HOME”)‘inC
Thismappingiscapturedthefirsttimethe[‘os‘]moduleisimported, typically
duringPythonstartupaspartofprocessing‘site.py‘. Changestotheenvironment
madeafterthistimearenotreflectedin[‘os.environ‘]exceptforchangesmadeby
modifying[‘os.environ‘]directly.
...
OnWindows,thekeysareconvertedtouppercase. Thisalsoapplieswhengetting,
setting,ordeletinganitem. Forexample,‘environ[’monty’]=’python’‘mapsthe
key‘’MONTY’‘tothevalue‘’python’‘.

PublishedasaconferencepaperatICLR2025
|             | Table13:                                          | AnexampleinLanguagedimension              |     |
| ----------- | ------------------------------------------------- | ----------------------------------------- | --- |
| CoreQuery   | Whatisdiabetes?                                   |                                           |     |
| Instructed1 | Tellmewhatdiabetesis.PleaseuseChinese.            |                                           |     |
| Instructed2 | Tellmetheanswertowhatisdiabetes.PleaseuseEnglish. |                                           |     |
| Reversed1   | Whatisdiabetes?                                   | PleaserespondinalanguageotherthanChinese. |     |
Reversed2 Whatisdiabetes? I’dratherhavearesponseinalanguageotherthanEnglish.
| Document1 | 糖尿病（拉丁语：diabetesmellitus，缩写为DM，简称diabetes）是一种代谢 |     |     |
| --------- | ----------------------------------------------- | --- | --- |
性疾病，它的特征是患者的血糖长期高于标准值。高血糖会造成俗称“三
多一少”的症状：多食、多饮、多尿及体重下降。对于第1型糖尿病，其症
状会在一个星期至一个月期间出现，而对于第2型糖尿病则较后出现。不
论是哪一种糖尿病，如果不进行治疗，可能会引发许多并发症。急性并
发症包括糖尿病酮酸血症与高渗透压高血糖非酮酸性昏迷；严重的长期并
发症则包括心血管疾病、中风、慢性肾脏病、糖尿病足、以及视网膜病变
等；其中糖尿病和心衰竭、慢性肾脏病有着较紧密的共病关系。
Document2 Diabetesisachronicdiseasethatoccurseitherwhenthepancreasdoesnotproduce
enoughinsulinorwhenthebodycannoteffectivelyusetheinsulinitproduces.
Table14: AnexampleinLengthdimension
| CoreQuery | Howmanycaloriesareinamartini? |     |     |
| --------- | ----------------------------- | --- | --- |
Instructed1 Howmanycaloriesareinamartini? Pleasegivemeasentenceanswer.
Instructed2 What’sthecaloriecountofamartini? I’dlikeaparagraphexplainingit.
Instructed3 Canyoutellmethecaloriesinamartini? Pleaseprovideadetailedarticle.
Reversed1 Howmanycaloriesareinamartini.Pleaseprovideadetailedresponse,notjusta
singlesentence.
Reversed2 Howmanycaloriesareinamartini.Pleaseavoidgivingmeaparagraphasyour
response.?
Reversed3 Howmanycaloriesareinamartini.Pleasedon’tstructureyouranswerasanarticle.
| Document1 | 2.25oz(67mL)Martini(extradry): |     | 140calories. |
| --------- | ------------------------------ | --- | ------------ |
Document2 TheamountofCaloriesinamartinicocktailcanvarybasedonhowyoumake
it. Amartinicocktailtechnicallyonlyhastwoingredients,vodkaandvermouth,
soCaloriecountdependsonyourproportions. GREYGOOSE®Vodkacontains
|     | 66Caloriesper30mlserving*. |     | TrymixingupourClassicDryVodkaMartini |
| --- | -------------------------- | --- | ------------------------------------ |
Cocktailrecipe.
| Document3 | VodkaMartiniCalories |     |     |
| --------- | -------------------- | --- | --- |
Dependingonthesizeofyourcocktail,andtheextrasyoumixin,oneservingofa
|     | vodkamartiniisapproximately202calories. |     | Vodkamartinicaloriescanbemuch |
| --- | --------------------------------------- | --- | ----------------------------- |
higherifthedrinkhasmorethanthetwobasicliquors.
|     | Tofigureoutthecaloriesinvodka... |     | 1teaspoonofFrenchvermouthhasapproxi- |
| --- | -------------------------------- | --- | ------------------------------------ |
mately7.8calories...

PublishedasaconferencepaperatICLR2025
Table15: AnexampleinSourcedimension
CoreQuery Effectiveexercisesforweightloss
Instructed1 What’sthebestwaytodoexercisesforweightlosseffectively? Pleaseprovidea
blogpostonthistopic.
Instructed2 HowcanIperformexerciseseffectivelyforweightloss? I’dlikeaforumposton
thissubject.
Instructed3 Tellmehowtodoeffectiveexercisesforweightloss. Givemesomethingfrom
NewsArticles.
Reversed1 Effectiveexercisesforweightloss.Pleaseprovidearesponsethatisnotfroma
blog.
Reversed2 Effectiveexercisesforweightloss.I’mlookingforananswerthat’snotbasedona
forumthread.
Reversed3 Effectiveexercisesforweightloss.Pleaseavoidusinganewsarticleasyoursource..
Document1 WhatAretheBestExercisesforWeightLoss?
May6,2024Blog
Losingweightcanbeachallengingjourney,butincorporatingexerciseintoyour
routinecanmakeasignificantdifference. Notonlydoesexercisehelpyouburn
calories,butitalsoboostsyourmetabolism,improvesyourmoodandincreases
youroverallhealthandwell-being.
Butwithsomanydifferenttypesofexercisesoutthere,itcanbeoverwhelmingto
figureoutwhichonesarethebestforweightloss.
HowtoExerciseforWeightLoss
Walkingexerciseforweightloss
Walkingisalow-impactexercisethatisperfectforbeginners...
Document2 superMario Milt:
Imyselfenjoygoingonlongwalks(anywherefrom30minutesto2hours). It’s
easyonthejoints,Icanlistentomusicorsticktomythoughts,andyougetfresh
airawayfrombeingcoopedupinagym. Itdefinitelyashelpedmetrimupsome
overtime.
Individual Ad 2701:
Ido1-2hoursofliftingadayhatecardiowellAfterliftingIdohowmuchshould
IwalkafterIliftlikewould20-30minutesworkI’mgainingmuscleandIcan
seethatmyarmsandchestarebiggerbutmybellyisgettingbiggeralsoIdidtry
eatinglesscaloriesbutidk.
Proudscobi:
Ifyouaregoingtochooseoneforweightloss,goforweightlifting.Itwillimprove
yourbodycomposition. Evenifyoudon’tloseweightyouwilllookbetter.
Document3 NBCHEALTHNEWS——Morningworkoutsmaybebetterforweightloss,study
finds. Peoplewhogottheirexerciseinbetween7a.m. and9a.m. hadlowerBMIs
thanthosewhooptedtoexerciselaterintheday. Ismorningthebesttimeofday
toexercise? ResearchpublishedTuesdayinthejournalObesityfindsthatearly
morningactivity—between7a.m. and9a.m. —couldhelpwithweightloss.
“Mycautioussuggestionfromthisstudyisthatifwechoosetoexerciseintheearly
morning,beforeweeat,wecan...

PublishedasaconferencepaperatICLR2025
Moreover,Figure4illustratesthedistributionofdataacrosssixdimensionsintheFollowIR,Instruc-
tIR,andInfoSearchdatasets. Thecharthighlightsthevaryingproportionsofquery-documentpairs
basedondimensionslikeAudience,Keyword,Language,Length,Source,andFormat.
|     | Audience | Keyword |     | Language   | Length | Source | Format     | Other |
| --- | -------- | ------- | --- | ---------- | ------ | ------ | ---------- | ----- |
|     | FollowIR |         |     | InstructIR |        |        | InFoSearch |       |
Figure 4: Comparison of the InfoSearch dataset with FollowIR and InstructIR in terms of data
distributionacrosssixdimensions.
| B   | EVALUATION | METRICS | ANALYSIS |     |     |     |     |     |
| --- | ---------- | ------- | -------- | --- | --- | --- | --- | --- |
To measure instruction-following performance for retrieval models is a challenge. Two metrics
werespecificallyproposedinpreviousstudiesforthispurpose: Robustness@k(Ohetal.,2024)and
p-MRR (Welleret al.,2024a). We argue thatneither ofthem effectivelyreflects trueinstruction-
followingperformance.
Robustness@k is designed to assess a model’s performance on the same query under different
instructions. Specifically,itgroupsinstancesofthesamequery,calculatestheminimumnDCG@k
scorewithineachgroup,andaveragesthegroupscorestogeneratetheoverallRobustness@kscore.
Let Q = {q ,q ,...,q } be a set of queries. For each query q , there are m distinct instruction
|     | 1 2 | n   |     |     |     | i   | i   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
variants{I ,I ,...,I },calculatetheminimumnDCG@kscoreacrossallitsinstruction:
|     | i1 i2 | imi |            |     |     |     |     |     |
| --- | ----- | --- | ---------- | --- | --- | --- | --- | --- |
|     |       |     | min-nDCG(q | )=  | min | s   |     | (7) |
|     |       |     |            | i   |     | ij  |     |     |
j∈(1,2,...,mi)
wheres representsthenDCG@kscoreforqueryq underinstructionI . Computetheoverall
|     | ij  |     |     |     | i   |     | ij  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
Robustness@kscoreastheaverageoftheseminimumscoresacrossallqueries:
n
1 (cid:88)
|     |     |     | Robustness@k | =   | min-nDCG(q | )   |     | (8) |
| --- | --- | --- | ------------ | --- | ---------- | --- | --- | --- |
i
n
i=1
However, the Robustness@k metric oversimplifies the evaluation of a model’s ability to follow
instructions. ⃝1 Evenifamodeldemonstratesstrongperformanceacrossthemajorityofqueries,
⃝2
asingleanomalouslylowscorecanreducetheoverallrobustnessscore. Furthermore,focusing
solelyonthelowestscoredisregardsvariationsinthemodel’sresponsestodifferentinstructions,thus
failingtocapturetheoverallperformancetrend.4
Asforp-MRR,itisbasedontheMRRmetricandquantifiesthemodel’sabilitytofollowinstructions
bycomparingtherankingsofrelevantdocumentsintheoriginalmodeandtheinstructionmode. The
followingformulaisappliedtocalculatethescoreforeachrelevantdocumentwithinaquery:

|     |     |        |     | M R R o g −1, | ifR        | >R     |     |     |
| --- | --- | ------ | --- | ------------- | ---------- | ------ | --- | --- |
|     |     |        |  M | R R           |            | og new |     |     |
|     |     | p-MRR= |     | n e w         |            |        |     | (9) |
|     |     |        | 1− | MRRnew,       | otherwise, |        |     |     |
MRRog
4Forinstance,thenDCG@kscoresforgroupAare{0.8,0.5,0.3,0.2},whilethoseforgroupBare{0.9,0.9,
0.9,0.2}.AlthoughgroupBexhibitsasignificantlybetteroverallperformance,Robustness@kassignsthesame
scoretobothgroups.

PublishedasaconferencepaperatICLR2025
whereMRRismeanreciprocalrank,R istherankofthedocumentintheoriginalretrievalmode,
og
and R is the new rank in the instruction mode. However, ⃝3 p-MRR fails to distinguish the
new
importanceofranking,neglectingtohighlightthecriticalrolethattopKresultsplayinretrieval.
⃝4 Moreover, the linear discount mechanism of p-MRR is insufficiently sensitive to changes in
higherrankings,makingitineffectiveincapturingsubtlemovementsatthetop. ⃝5 Lastly,p-MRR
demonstrateslimitationswhenaddressingspecialcasesandextremeperformances.5
C PROMPT FOR LIST-WISE RERANKING MODELS
Table 16: Prompt for List-wise Reranking Models. The input consists of a list of documents or
passages,andthemodelispromptedtoreturnarankedlistofdocumentIDsbasedontheirrelevance
tothequery.
TASK PromptTemplate
Rank <|system|>
YouareRankGPT,anintelligentassistantthatrankspassagesbasedontheirrelevanceto
aquery.
<|user|>
Iwillprovideyouwith{num}passages,eachindicatedbyanumberidentifier[]. Rank
thepassagesbasedontheirrelevancetothequery: {query}.
[1]{passage1}
[2]{passage2}
...
[num]{passage{num}}
SearchQuery: {query}.
Rank the {num} passages above based on their relevance to the search query. The
passages should be listed in descending order using identifiers. The most relevant
passagesshouldbelistedfirst. Theoutputformatshouldbe[]>[],e.g.,[1]>[2]. Only
respondwiththerankingresults,donotsayanywordorexplain.
<|assistant|>
ModelGeneration: [9]>[4]>[20]>... >[13]
5Forexample,theperformanceofmodel1isR =10andR =5,yieldingap-MRRof-0.5,while
og new
model2’sperformanceisR =100andR =50,resultinginap-MRRof-0.5. Althoughbothmodels
og new
receivethesamescore,itisevidentthatmodel1hasagreaterimpactontheretrievalresults.

PublishedasaconferencepaperatICLR2025
| D SAMPLING | WISE SCORE | REWARD | COMPONENT |     |     |
| ---------- | ---------- | ------ | --------- | --- | --- |
OF
𝑅(cid:3042)(cid:3045)(cid:3036)
|     | 22 21 20 19             | 18 17 16 15 14 13                   | 12 11 10 9 8 7                      | 6 5 4 3 2 1  𝑅(cid:3036)(cid:3041)(cid:3046) |     |
| --- | ----------------------- | ----------------------------------- | ----------------------------------- | -------------------------------------------- | --- |
|     | 0.100 0.100 0.782 0.788 | 0.794 0.800 0.806 0.813 0.820 0.827 | 0.834 0.842 0.850 0.859 0.868 0.878 | 0.888 0.900 0.913 0.929 0.950 1.000          |     |
1
|     | 0.100 0.100 0.557 0.561 | 0.566 0.570 0.575 0.580 0.585 0.590 | 0.595 0.601 0.607 0.614 0.621 0.628 | 0.636 0.646 0.657 0.672 0.000 |     |
| --- | ----------------------- | ----------------------------------- | ----------------------------------- | ----------------------------- | --- |
2
|     | 0.100 0.100 0.458 0.462 | 0.466 0.469 0.473 0.477 0.482 0.486 | 0.491 0.496 0.501 0.507 0.513 0.520 | 0.527 0.537 0.548 0.000 |     |
| --- | ----------------------- | ----------------------------------- | ----------------------------------- | ----------------------- | --- |
3
|     | 0.100 0.100 0.400 0.403 | 0.406 0.410 0.413 0.417 0.421 0.425 | 0.429 0.434 0.439 0.444 0.450 0.457 | 0.465 0.475 0.000 |     |
| --- | ----------------------- | ----------------------------------- | ----------------------------------- | ----------------- | --- |
4
|     | 0.100 0.100 0.361 0.364 | 0.367 0.370 0.373 0.377 0.380 0.384 | 0.388 0.392 0.397 0.402 0.408 0.416 | 0.425 0.000 |     |
| --- | ----------------------- | ----------------------------------- | ----------------------------------- | ----------- | --- |
5
|     | 0.100 0.100 0.332 0.335 | 0.338 0.341 0.344 0.347 0.351 0.354 | 0.358 0.363 0.367 0.373 0.379 0.388 | 0.000 |     |
| --- | ----------------------- | ----------------------------------- | ----------------------------------- | ----- | --- |
6
|     | 0.100 0.100 0.310 0.312 | 0.315 0.318 0.321 0.325 0.328 0.332 | 0.336 0.340 0.345 0.351 0.359 0.000 |     |     |
| --- | ----------------------- | ----------------------------------- | ----------------------------------- | --- | --- |
7
|     | 0.100 0.100 0.292 0.295 | 0.298 0.301 0.304 0.307 0.310 0.314 | 0.318 0.323 0.329 0.336 0.000 |     |     |
| --- | ----------------------- | ----------------------------------- | ----------------------------- | --- | --- |
8
|     | 0.100 0.100 0.278 0.281 | 0.283 0.286 0.289 0.293 0.296 0.300 | 0.304 0.310 0.317 0.000 |     |     |
| --- | ----------------------- | ----------------------------------- | ----------------------- | --- | --- |
9
|     | 0.100 0.100 0.266 0.269 | 0.272 0.274 0.277 0.281 0.285 0.289 | 0.294 0.300 0.000 |     |     |
| --- | ----------------------- | ----------------------------------- | ----------------- | --- | --- |
10
|     | 0.100 0.100 0.256 0.259 | 0.262 0.265 0.268 0.271 0.275 0.280 | 0.286 0.000 | 11  |     |
| --- | ----------------------- | ----------------------------------- | ----------- | --- | --- |
|     | 0.100 0.100 0.248 0.250 | 0.253 0.256 0.260 0.264 0.268 0.274 | 0.000       |     |     |
12
|     | 0.100 0.100 0.241 0.243 | 0.246 0.250 0.253 0.258 0.263 0.000 |     |     |     |
| --- | ----------------------- | ----------------------------------- | --- | --- | --- |
13
|     | 0.100 0.100 0.235 0.237 | 0.241 0.244 0.248 0.254 0.000 |     | 14  |     |
| --- | ----------------------- | ----------------------------- | --- | --- | --- |
|     | 0.100 0.100 0.229 0.232 | 0.236 0.240 0.245 0.000       |     |     |     |
15
|     | 0.100 0.100 0.225 0.228 | 0.232 0.238 0.000 |     |     |     |
| --- | ----------------------- | ----------------- | --- | --- | --- |
16
|     | 0.100 0.100 0.222 0.225 | 0.230 0.000 |     |     |     |
| --- | ----------------------- | ----------- | --- | --- | --- |
17
0.100 0.100 0.219 0.224 0.000
18
0.100 0.100 0.218 0.000
19
0.100 0.100 0.000
20
|     | 0.100 0.000 |     |     | 21  |     |
| --- | ----------- | --- | --- | --- | --- |
0.000
22
|                | Figure5: | Heatmapoftherewardscomponent |     |                 |         |
| -------------- | -------- | ---------------------------- | --- | --------------- | ------- |
| E THE COMPLETE | RESULTS  | OF EVALUATING                |     | WITH INFOSEARCH | DATASET |
Table 17, Table 18, Table 19, Table 20, Table 21, and Table 22 show all the results of the 15
retrievalmodelsinInfoSearchdataset. OriindicatesmodelsevaluateinOriginalmode. Insindicates
modelsevaluateinInstructedmode. RevindicatesmodelsevaluateinReverselyinstructedmode. Act.
indicatestheactualperformanceofthemodelandidealindicatestheidealperformance.Per.indicates

PublishedasaconferencepaperatICLR2025
howfartheactualperformanceisfromtheidealperformanceasaproportionoftheidealperformance.
Alowerpercentageindicatesthattheactualperformanceisclosertotheidealperformance,whilea
higherpercentageindicatesagreaterdeviationfromtheidealperformance. Thecalculationformula
isPer.= ideal−actual.
ideal
Table17: AudienceResults
Audience-(Layman,Expert)
|       | nDCG@10   |      | MRR@1          | WISE         |           |
| ----- | --------- | ---- | -------------- | ------------ | --------- |
| Model |           |      |                |              | SICR↑     |
|       | Or Ch     | Re   | Or Ch Re       | Act.↑ Ideal↑ | Per.↓     |
| BM25  | 46.1 38.7 | 36.4 | 21.0 11.9 13.3 | -3.0 65.9    | 104.6 0.0 |
DenseRetrieval
Bge-Large-v1.5 48.6 38.1 37.6 22.9 12.9 11.9 -16.8 67.5 124.9 0.5
E5-Large-v2 53.9 45.3 42.6 32.4 16.7 16.2 -15.6 71.7 121.7 1.4
Instructor-XL 48.3 30.1 31.2 29.5 8.6 10.0 -27.7 64.6 142.9 5.7
Mistral-ins-v0.2 31.1 35.6 37.5 20.0 17.1 17.1 -35.8 40.6 188.3 0.0
E5-Mistral-ins 78.9 63.3 64.3 72.4 34.8 35.2 -7.3 86.1 108.5 2.9
| GritLM    | 56.2 56.7 | 57.1 | 41.9 31.4 30.0 | -3.4 70.2  | 104.9 11.4 |
| --------- | --------- | ---- | -------------- | ---------- | ---------- |
| GTE-Qwen2 | 56.4 57.0 | 57.3 | 46.7 35.2 35.2 | -34.0 65.3 | 152.0 0.0  |
SFR-Embedding-2-R 63.2 51.6 52.0 41.9 24.8 22.9 -7.8 79.2 109.9 2.9
NV-Embed-v2 65.3 47.6 47.5 44.8 17.6 17.1 -9.8 80.5 112.2 2.4
Point-wiseReranking
Mistral-ins-v0.2 75.8 60.9 63.6 62.9 28.1 35.2 -8.9 85.0 110.4 1.4
| Llama-3.1 | 79.9 65.1 | 67.4 | 68.6 36.7 41.4 | -6.2 88.4 | 107.0 6.2 |
| --------- | --------- | ---- | -------------- | --------- | --------- |
| FollowIR  | 76.9 64.9 | 63.6 | 69.5 35.7 35.2 | -2.3 85.7 | 102.6 3.3 |
List-wiseReranking
Mistral-ins-v0.2 68.7 58.9 58.6 51.4 29.0 28.6 -6.3 81.0 107.8 10.5
Zephyr-beta 77.0 62.1 62.6 71.4 35.2 37.6 -2.7 84.8 103.2 1.0
RankVicuna-v1 62.2 52.5 51.2 50.5 27.1 25.7 -2.5 75.1 103.3 5.2
RankZephyr-v1 71.0 58.9 59.2 56.2 31.0 30.0 7.4 82.6 91.1 4.3
| GPT-4o | 87.7 72.5 | 72.6 | 88.6 48.6 48.6 | 7.4 95.9 | 92.2 15.2 |
| ------ | --------- | ---- | -------------- | -------- | --------- |

PublishedasaconferencepaperatICLR2025
Table18: KeywordResults
Keywords-(Include[keywords])
|       | nDCG@10   |      | MRR@1          | WISE         |           |
| ----- | --------- | ---- | -------------- | ------------ | --------- |
| Model |           |      |                |              | SICR↑     |
|       | Or Ch     | Re   | Or Ch Re       | Act.↑ Ideal↑ | Per.↓     |
| BM25  | 70.4 70.0 | 54.1 | 64.5 45.3 24.7 | -42.1 77.7   | 154.2 0.0 |
DenseRetrieval
Bge-Large-v1.5 46.2 39.7 29.4 25.8 11.8 11.5 -38.2 66.2 157.8 0.0
E5-Large-v2 60.1 70.6 45.0 43.2 46.7 16.0 -38.3 75.6 150.7 0.7
Instructor-XL 68.5 48.7 38.4 56.8 19.9 14.6 -34.7 79.4 143.7 2.1
Mistral-ins-v0.2 31.7 28.5 37.0 30.7 7.3 19.9 -67.8 36.0 288.4 0.0
E5-Mistral-ins 72.3 79.5 71.8 60.6 57.1 33.1 -44.5 80.7 155.2 0.0
| GritLM    | 85.9 79.4 | 67.2 | 89.2 58.2 46.0 | 6.8 86.6   | 92.2 11.8 |
| --------- | --------- | ---- | -------------- | ---------- | --------- |
| GTE-Qwen2 | 58.9 43.5 | 49.3 | 58.2 18.1 32.8 | -36.5 64.9 | 156.2 0.0 |
SFR-Embedding-2-R 47.3 64.5 47.7 30.7 38.3 24.4 -45.9 65.4 170.3 1.0
NV-Embed-v2 61.5 61.4 40.2 49.8 34.8 17.1 -27.7 74.7 137.1 0.7
Point-wiseReranking
Mistral-ins-v0.2 39.9 63.6 38.0 16.7 40.1 16.0 34.5 62.4 44.7 28.6
| Llama-3.1 | 61.7 76.9 | 48.4 | 48.4 54.0 28.2 | 38.7 74.2 | 47.8 38.3 |
| --------- | --------- | ---- | -------------- | --------- | --------- |
| FollowIR  | 51.2 78.1 | 45.7 | 34.1 59.9 25.8 | 47.7 68.7 | 30.6 27.2 |
List-wiseReranking
Mistral-ins-v0.2 67.4 79.3 43.8 59.2 64.1 27.2 46.0 76.8 40.1 59.2
Zephyr-beta 68.9 65.2 47.8 59.9 47.0 32.8 14.1 77.0 81.7 27.5
RankVicuna-v1 66.8 75.6 51.9 65.2 57.8 32.1 -8.5 75.7 111.2 10.5
RankZephyr-v1 72.6 77.3 52.0 79.1 60.6 34.5 53.9 92.3 41.6 42.5
| GPT-4o | 71.8 78.8 | 61.9 | 66.2 70.7 51.6 | 63.0 86.0 | 26.7 60.6 |
| ------ | --------- | ---- | -------------- | --------- | --------- |

PublishedasaconferencepaperatICLR2025
Table19: FormatResults
Format-(StackoverflowPost,CodeSnippet,OfficialManual
|       | nDCG@10   |      | MRR@1        | WISE         |           |
| ----- | --------- | ---- | ------------ | ------------ | --------- |
| Model |           |      |              |              | SICR↑     |
|       | Or Ch     | Re   | Or Ch Re     | Act.↑ Ideal↑ | Per.↓     |
| BM25  | 22.6 15.3 | 19.1 | 16.0 4.7 8.3 | -2.8 30.7    | 109.0 0.0 |
DenseRetrieval
Bge-Large-v1.5 58.0 25.9 31.4 46.0 5.3 11.3 -42.1 65.8 163.9 0.3
E5-Large-v2 59.3 44.9 52.0 44.0 20.3 36.7 -15.5 68.9 122.5 0.0
Instructor-XL 64.2 35.7 40.6 54.0 11.7 20.3 -30.5 72.5 142.1 0.3
| Mistral-ins-v0.2 | 2.4 3.2 | 4.3 | 0.0 1.0 1.3 | -29.7 5.6 | 630.7 0.0 |
| ---------------- | ------- | --- | ----------- | --------- | --------- |
E5-Mistral-ins 72.2 46.7 58.9 72.0 17.7 37.0 -19.9 75.9 126.3 0.0
| GritLM    | 45.5 48.4 | 53.8 | 31.0 21.3 38.7 | -36.0 54.6 | 165.9 1.3 |
| --------- | --------- | ---- | -------------- | ---------- | --------- |
| GTE-Qwen2 | 14.3 14.6 | 19.0 | 13.0 5.3 12.3  | -44.3 18.2 | 343.6 0.0 |
SFR-Embedding-2-R 75.4 53.0 63.1 76.0 23.7 46.3 -13.0 78.5 116.5 2.0
NV-Embed-v2 67.5 41.5 53.1 59.0 15.3 30.7 -18.1 73.4 124.7 1.0
Point-wiseReranking
Mistral-ins-v0.2 62.2 50.8 61.7 43.0 21.3 35.7 -9.3 74.2 112.6 2.7
| Llama-3.1 | 68.0 51.2 | 59.5 | 58.0 18.3 32.7 | -9.5 77.3 | 112.3 10.0 |
| --------- | --------- | ---- | -------------- | --------- | ---------- |
| FollowIR  | 72.2 54.9 | 68.5 | 62.0 23.3 50.0 | -2.3 79.7 | 102.9 7.0  |
List-wiseReranking
Mistral-ins-v0.2 69.1 50.8 58.3 69.0 24.3 42.3 -6.6 74.6 108.8 9.7
Zephyr-beta 48.8 35.2 42.3 51.0 15.3 36.7 -13.9 58.3 123.8 8.0
RankVicuna-v1 44.4 33.3 41.3 32.0 12.7 25.3 -9.8 56.7 117.2 3.3
RankZephyr-v1 73.0 52.6 63.5 61.0 20.3 41.7 7.8 81.2 90.4 1.0
| GPT-4o | 78.7 59.4 | 67.7 | 84.0 32.0 52.3 | 21.9 94.3 | 76.8 11.3 |
| ------ | --------- | ---- | -------------- | --------- | --------- |

PublishedasaconferencepaperatICLR2025
Table20: LanguageResults
Language-(Chinese,English)
|       | nDCG@10   |      | MRR@1          | WISE         |           |
| ----- | --------- | ---- | -------------- | ------------ | --------- |
| Model |           |      |                |              | SICR↑     |
|       | Or Ch     | Re   | Or Ch Re       | Act.↑ Ideal↑ | Per.↓     |
| BM25  | 36.1 30.3 | 28.8 | 28.0 14.0 13.0 | -7.2 43.7    | 116.5 0.0 |
DenseRetrieval
Bge-Large-v1.5 42.7 36.1 32.0 38.0 19.0 13.5 -20.7 51.5 140.1 2.0
E5-Large-v2 52.4 50.2 44.6 58.0 35.5 33.5 -25.3 56.7 144.7 0.5
Instructor-XL 47.6 37.7 35.0 54.0 25.0 19.5 -20.5 50.0 141.1 4.0
Mistral-ins-v0.2 20.7 30.0 29.5 19.0 21.0 19.0 -31.9 26.5 220.6 0.0
E5-Mistral-ins 81.5 73.4 62.4 80.0 48.5 40.0 0.1 87.4 99.9 10.5
| GritLM    | 82.6 81.0 | 75.9 | 78.0 57.0 48.0 | -6.7 87.7  | 107.7 4.5 |
| --------- | --------- | ---- | -------------- | ---------- | --------- |
| GTE-Qwen2 | 38.5 36.8 | 37.9 | 43.0 27.0 28.5 | -18.0 41.1 | 143.7 0.0 |
SFR-Embedding-2-R 81.5 81.7 64.3 77.0 61.0 31.5 -15.4 88.1 117.4 1.0
NV-Embed-v2 68.3 67.4 59.6 72.0 39.0 33.5 -7.3 76.8 109.5 3.5
Point-wiseReranking
Mistral-ins-v0.2 58.9 63.3 60.8 32.0 38.5 32.0 21.4 76.6 72.0 6.5
| Llama-3.1 | 67.9 71.9 | 64.2 | 61.0 44.5 36.5 | 29.0 79.4 | 63.5 22.0 |
| --------- | --------- | ---- | -------------- | --------- | --------- |
| FollowIR  | 68.4 70.6 | 64.7 | 54.0 48.0 37.0 | 20.9 81.2 | 74.3 19.5 |
List-wiseReranking
Mistral-ins-v0.2 73.7 70.7 63.5 69.0 49.5 38.5 7.6 81.9 90.7 23.0
Zephyr-beta 70.9 58.7 58.1 68.0 38.5 37.5 -6.9 79.3 108.7 10.5
RankVicuna-v1 63.4 55.5 53.2 54.0 29.5 29.5 -11.8 76.3 115.5 4.5
RankZephyr-v1 79.5 66.6 66.9 69.5 38.5 37.5 10.6 88.0 87.9 5.5
| GPT-4o | 83.2 86.2 | 82.2 | 83.0 75.5 65.5 | 53.1 91.3 | 41.8 55.5 |
| ------ | --------- | ---- | -------------- | --------- | --------- |

PublishedasaconferencepaperatICLR2025
Table21: LengthResults
Length-(Sentence,Paragraph,Article)
|       | nDCG@10   |      | MRR@1          | WISE         |           |
| ----- | --------- | ---- | -------------- | ------------ | --------- |
| Model |           |      |                |              | SICR↑     |
|       | Or Ch     | Re   | Or Ch Re       | Act.↑ Ideal↑ | Per.↓     |
| BM25  | 63.3 43.4 | 54.1 | 64.0 19.3 40.7 | -7.5 71.5    | 110.4 0.0 |
DenseRetrieval
Bge-Large-v1.5 62.7 35.1 42.0 46.0 9.0 15.7 -28.6 74.3 138.4 4.7
E5-Large-v2 73.9 52.6 63.0 66.0 26.3 48.0 -21.6 80.3 126.9 2.7
Instructor-XL 75.7 38.1 47.3 74.0 12.3 25.3 -35.5 81.3 143.7 1.7
Mistral-ins-v0.2 11.8 22.6 27.6 13.0 11.7 25.0 -66.6 13.7 586.2 0.0
E5-Mistral-ins 86.2 64.2 76.2 92.0 33.0 60.3 -13.4 86.0 115.6 0.0
| GritLM    | 74.0 65.4 | 76.6 | 76.0 36.0 61.7 | -25.8 79.0 | 132.6 0.3 |
| --------- | --------- | ---- | -------------- | ---------- | --------- |
| GTE-Qwen2 | 34.4 47.3 | 55.0 | 32.0 22.3 42.7 | -56.4 40.7 | 238.6 0.3 |
SFR-Embedding-2-R 75.7 59.1 70.7 70.0 27.7 53.3 -13.5 81.7 116.5 1.0
NV-Embed-v2 81.9 55.3 65.6 83.0 26.7 48.0 -9.3 84.6 111.0 0.3
Point-wiseReranking
Mistral-ins-v0.2 60.3 52.3 60.2 42.0 24.0 38.0 -12.6 74.4 116.9 4.7
| Llama-3.1 | 84.4 64.1 | 73.0 | 86.0 36.7 56.0 | -5.9 87.6 | 106.8 2.7 |
| --------- | --------- | ---- | -------------- | --------- | --------- |
| FollowIR  | 80.3 61.9 | 71.5 | 80.0 35.0 55.7 | -2.6 84.6 | 103.0 1.7 |
List-wiseReranking
Mistral-ins-v0.2 88.7 66.0 76.8 92.0 38.7 64.7 -1.9 89.1 102.2 8.0
Zephyr-beta 85.8 61.5 74.6 93.0 31.7 63.0 -5.7 86.5 106.6 2.0
RankVicuna-v1 84.8 61.8 74.1 93.0 32.7 60.7 -4.3 85.4 105.0 2.3
RankZephyr-v1 86.1 63.5 75.7 90.0 34.0 59.7 7.8 88.2 91.1 4.3
| GPT-4o | 89.1 68.8 | 79.0 | 95.0 42.0 67.3 | 10.2 94.4 | 89.2 10.3 |
| ------ | --------- | ---- | -------------- | --------- | --------- |

PublishedasaconferencepaperatICLR2025
Table22: SourceResults
Source-(Blog,ForumPost,NewsArticle)
|       | nDCG@10   |      | MRR@1          | WISE         |           |
| ----- | --------- | ---- | -------------- | ------------ | --------- |
| Model |           |      |                |              | SICR↑     |
|       | Or Ch     | Re   | Or Ch Re       | Act.↑ Ideal↑ | Per.↓     |
| BM25  | 45.8 37.0 | 38.3 | 25.0 15.7 15.7 | -9.4 63.4    | 114.9 0.0 |
DenseRetrieval
Bge-Large-v1.5 59.1 34.6 36.8 49.0 12.3 16.7 -30.7 71.4 143.0 2.3
E5-Large-v2 61.1 48.6 52.1 49.0 22.7 29.0 -23.2 72.6 132.0 1.3
Instructor-XL 70.5 40.0 43.7 66.0 13.0 15.3 -29.8 77.7 138.3 4.0
Mistral-ins-v0.2 19.9 32.9 39.2 18.0 12.0 24.0 -63.3 23.9 364.7 0.0
E5-Mistral-ins 76.2 58.9 62.4 70.0 30.7 36.0 -13.0 82.5 115.8 3.3
| GritLM    | 80.3 66.3 | 67.5 | 76.0 40.7 46.0 | -1.5 84.4  | 101.8 11.7 |
| --------- | --------- | ---- | -------------- | ---------- | ---------- |
| GTE-Qwen2 | 58.2 59.2 | 72.4 | 55.0 29.0 53.0 | -44.6 63.6 | 170.1 0.0  |
SFR-Embedding-2-R 78.9 63.1 62.6 74.0 37.0 35.3 -13.2 84.0 115.7 4.7
NV-Embed-v2 71.3 53.8 47.5 67.0 29.7 23.3 -8.7 78.5 111.1 9.0
Point-wiseReranking
Mistral-ins-v0.2 71.5 59.7 69.7 56.0 24.3 48.0 -0.3 81.0 100.4 4.7
| Llama-3.1 | 84.9 71.4 | 79.9 | 91.0 46.0 75.7 | 40.2 84.8 | 52.6 36.7 |
| --------- | --------- | ---- | -------------- | --------- | --------- |
| FollowIR  | 81.9 67.4 | 79.3 | 74.0 35.7 65.0 | 18.8 86.1 | 78.2 16.3 |
List-wiseReranking
Mistral-ins-v0.2 77.7 60.8 68.6 78.0 34.3 57.0 10.0 81.0 87.6 21.7
Zephyr-beta 70.4 52.9 62.8 65.0 23.7 44.0 -3.9 78.2 105.0 3.0
RankVicuna-v1 68.7 52.7 59.5 74.0 30.3 50.0 -2.2 72.6 103.0 8.0
RankZephyr-v1 82.8 61.8 71.0 85.0 33.7 56.0 -0.3 83.5 100.4 5.3
| GPT-4o | 89.4 79.5 | 82.0 | 93.0 65.7 77.3 | 45.2 95.5 | 52.7 39.7 |
| ------ | --------- | ---- | -------------- | --------- | --------- |
