A Comprehensive Taxonomy of Negation for NLP and Neural Retrievers
RoxanaPetcu,SamarthBhargav,MaartendeRijke,EvangelosKanoulas
UniversityofAmsterdam,TheNetherlands
|     |     | {r.m.petcu, | s.bhargav, |     | m.derijke, |                                            | e.kanoulas}@uva.nl |     |     |     |     |     |
| --- | --- | ----------- | ---------- | --- | ---------- | ------------------------------------------ | ------------------ | --- | --- | --- | --- | --- |
|     |     | Abstract    |            |     |            | guaranteedtobepresentinthetrainingregimeof |                    |     |     |     |     |     |
anymodel,takesdifferentformsdependingonthe
Understandingandsolvingcomplexreasoning
5202 tcO 41  ]LC.sc[  3v73322.7052:viXra task at hand. Human comprehension of negation
| tasks | is vital | for addressing | the | information |     |     |     |     |     |     |     |     |
| ----- | -------- | -------------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
comesasaresultofunderstandinglinguistic,mor-
| needsofauser. |     | Althoughdenseneuralmodels |     |     |     |     |     |     |     |     |     |     |
| ------------- | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
phological,andsyntacticconstructionalongwith
| learncontextualisedembeddings,      |     |     |     | theyunder- |     |        |      |             |     |          |      |         |
| ----------------------------------- | --- | --- | --- | ---------- | --- | ------ | ---- | ----------- | --- | -------- | ---- | ------- |
|                                     |     |     |     |            |     | verbal | cues | (as defined | in  | Appendix | A.1) | and fa- |
| performonqueriescontainingnegation. |     |     |     | Toun-      |     |        |      |             |     |          |      |         |
derstandthisphenomenon,westudynegation cialexpressions(Zuanazzietal.,2023). However,
intraditionalneuralinformationretrievaland this multifaceted linguistic phenomenon is often
LLM-basedmodels. We(1)introduceataxon- reducedtoabinarydescriptioninlanguageprocess-
omyofnegationthatderivesfromphilosophi-
|     |     |     |     |     |     | ingsystems: |     | Doesnegationexistornotinaspecific |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ----------- | --- | --------------------------------- | --- | --- | --- | --- |
cal,linguistic,andlogicaldefinitions;(2)gener-
dataset(Welleretal.,2024;Zhangetal.,2024a),
atetwobenchmarkdatasetsthatcanbeusedto
|     |     |     |     |     |     | and | is it encoded | or  | not | by a model | (Ravichan- |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | ---------- | ---------- | --- |
evaluatetheperformanceofneuralinformation
|     |     |     |     |     |     | der et | al., 2022). | Addressing |     | these | discrepancies |     |
| --- | --- | --- | --- | --- | --- | ------ | ----------- | ---------- | --- | ----- | ------------- | --- |
retrievalmodelsandtofine-tunemodelsfora
morerobustperformanceonnegation;and(3) betweenhumanandsystemunderstandingofnega-
proposealogic-basedclassificationmechanism tion,weaskthefollowingresearchquestions:
thatcanbeusedtoanalyzetheperformanceof (RQ1) Canwedesignacomprehensivetaxonomy
| retrievalmodelsonexistingdatasets. |     |     |     | Ourtax- |     |     |     |     |     |     |     |     |
| ---------------------------------- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
fornegation?
| onomy | produces | a balanced | data | distribution |     |       |                                     |     |     |     |     |     |
| ----- | -------- | ---------- | ---- | ------------ | --- | ----- | ----------------------------------- | --- | --- | --- | --- | --- |
|       |          |            |      |              |     | (RQ2) | Howcanthistaxonomybeappliedtogener- |     |     |     |     |     |
overnegationtypes,providingabettertraining
ateamorecompleteandbalanceddataset?
| setup         | that leads | to faster                | convergence | on  | the |       |         |        |      |       |             |     |
| ------------- | ---------- | ------------------------ | ----------- | --- | --- | ----- | ------- | ------ | ---- | ----- | ----------- | --- |
|               |            |                          |             |     |     | (RQ3) | In what | manner | does | model | performance |     |
| NevIRdataset. |            | Moreover,weproposeaclas- |             |     |     |       |         |        |      |       |             |     |
sificationschemathatrevealsthecoverageof differ when fine-tuned on the taxonomy-
drivendataset
negationtypesinexistingdatasets,offeringin-
sightsintothefactorsthatmightaffectthegen- versuspriorexistingdatasets?
| eralization | of  | fine-tuned | models | on negation. |     |       |     |          |          |     |         |        |
| ----------- | --- | ---------- | ------ | ------------ | --- | ----- | --- | -------- | -------- | --- | ------- | ------ |
|             |     |            |        |              |     | (RQ4) | How | can this | taxonomy | be  | used to | under- |
OurcodeispubliclyavailableonGitHub1,and standwhymodelsunderperformonexisting
thedatasetsareavailableonHuggingFace23.
negationdatasets?
RQ1aimstobringtogetherresearchfromthelin-
1 Introduction
|     |     |     |     |     |     | guistic | literature | in  | a taxonomy | on  | negation. | We  |
| --- | --- | --- | --- | --- | --- | ------- | ---------- | --- | ---------- | --- | --------- | --- |
Akeyfactorcontributingtoaccuraterelevancein designourtaxonomytobeexhaustive,withnoover-
neural information retrieval (IR) systems, LLM- lap,andrelevanttoIRtasks. ToaddressRQ2,we
| based re-rankers, |     | and retrieval | augmented |     | gener- |     |     |     |     |     |     |     |
| ----------------- | --- | ------------- | --------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
proposetwosyntheticallygenerateddatasetsthat
ation(RAG)isacquiringlanguageunderstanding coverallproposednegationtypes. Figure1illus-
capabilities through pre-training (Hosseini et al., tratesthetaskalongsidethedatatyperepresented
2021). Despitetheirextensivetrainingsetups,these inourdatasets. RQ3analyzestheperformanceof
modelsshowpersistentdifficultyinhandlingnega-
|     |     |     |     |     |     | neural | IR models, | providing |     | insight | into | the gap |
| --- | --- | --- | --- | --- | --- | ------ | ---------- | --------- | --- | ------- | ---- | ------- |
tion (McKenzie et al., 2024), both in spoken and betweenhumanunderstandingandLLMencoding
writtenlanguage(Ortegaetal.,2016). Negationis of negation. RQ4 connects the taxonomy to for-
| linguistically | a   | complex phenomenon |     | that, | while |     |     |     |     |     |     |     |
| -------------- | --- | ------------------ | --- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- |
malizationsthatcanbeusedasdataclassification
|     |     |     |     |     |     | mechanisms, |     | allowing | to  | study existing | datasets |     |
| --- | --- | --- | --- | --- | --- | ----------- | --- | -------- | --- | -------------- | -------- | --- |
1github.com/RoxanaPetcu/taxonomy-negation
2gpt4o-negation-controlled andidentifyreasonswhyfine-tuningdoesnotguar-
| 3gpt4o-negation-free |     |     |     |     |     | anteeaperformanceboost. |     |     |     |     |     |     |
| -------------------- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- |

Doc 1 Doc 2
Plants absorb light primarily using the pigment Chlorophyll in green plants absorbs light from
chlorophyll. The green part of the light the blue and red spectrums of light. These
spectrum is not absorbed but is reflected, absorbed wavelengths drive the photosynthetic
which is the reason that most plants have no reactions by energizing electrons in the
green color. chlorophyll molecules.
Ranked list Ranked list
Similarity Doc 2
1 Doc 1 1 Doc 2
W ab h s ic o h r b li e g d h t b s y p c e h c l t o r r u o m ph w y a ll v i e n l e g n re g e th n s p a l r a e n t n s o ? t Doc 1 2 Doc 2 2 Doc 1
Similarity Ranked list Ranked list
Doc 2
1 Doc 1 1 Doc 2
Which light spectrum wavelengths are
absorbed by chlorophyll in green plants? Doc 1 2 Doc 2 2 Doc 1
Figure1: ExampleinstancefromourFreeGenerationdatasetforsententialnegation. Doc1isapassageretrieved
fromanexistingWikipediaarticle;Doc2isaminimallyeditedcounterfactualwhosetruthvalueisflipped. Thetask
ispairwiseranking. Giventwoqueriesthatonlydifferinthepresenceofnegation,theretrievalmodelmustrankthe
correspondingdocumenthigher. Themodelsucceedsifitranksthecorrectdocumenthigherforbothqueries. There
isa25%randomchanceinpairwiseaccuracy.
2 Motivation etal.,2019;Merchantetal.,2020). Modelsensi-
tivitytoparameteradjustmentsisparticularlyno-
Negationhasalonghistoryin(computational)
ticeableininformationretrievalsettings. Thishas
linguistics. Thestudyofoppositionanditsexpres-
beenobservedintraditionalBERT-basedarchitec-
sionintheformofnegationisaphenomenonthat
tures (Gerritse et al., 2022) and LLMs (Soudani
hasbeendebatedby,andprovokedinterestfromlin-
etal.,2024a). Althoughthisbehaviorcanbemiti-
guists,logicians,metaphysicians,andphilosophers
gatedbyfreezingthemodelparametersandadding
(Seiver,1944;Horn,1989;Kunen,1987;Halpern
alanguagemodelheadthatisfine-tunedonanew
and Pearl, 2005). It is a highly complex expres-
dataset(Huangetal.,2022;Linetal.,2022),this
sion of thought given its apparent simple form
method restricts the capabilities the model can
(Horn, 1989). Other challenges are imposed by
learn. Welleretal.(2024)showsthatfine-tuningon
theambiguityofthenegationscope(Atlas,1977),
theirproposeddataset(NevIR)leadstoanoticeable
andpragmaticinferencesinconversationalsettings
declineinMSMarcogeneralizationperformance.
(SchlöderandFernández,2015).
Representations of negation. Another explana-
Proper treatment of negation is essential. Un-
tion for models under-performing on negation is
derstandingnegationisvitalforretrievalmodelsto
under-representation of negation in crawled pre-
providethecorrectinformationtotheuser. More-
training datasets (Hossain et al., 2020). An im-
over, handling negation is vital to ensure that the
propertrainingcanalsobecausedbythetraining
retrievedgenerationsareacorrectresponsetothe
objective. Whilecontrastivelosspushesdifferent
userquery,sincegeneratedanswersareparticularly
content to be distant in the representation space,
difficult to verify, as they cannot be grounded in
twonegatedstatementsarecloseincontentwhile
establishedevidence(Wangetal.,2024). Equally
conveying opposite information. (Hosseini et al.,
important is ensuring that RAG systems respect
2021;NojiandTakamura,2020)addresstheprob-
user-specifiednegationandavoidretrievinginfor-
lem of misalignment between training objective
mationtheuserexplicitlydoesnotsearchfor.
andsemanticsbyproposingan‘unlikelihood’loss
Fine-tuningonnegationdatasets. Onecouldar-
functiontopre-trainBERTonfactuallyincorrect
gue that this problem can be mitigated through
statementswithnegationcues. Recently,(Krasakis
fine-tuning (Dolci, 2022). However, catastrophic
etal.,2025)constructedcompositionalqueryrep-
forgettingoccurswhenamodelisfine-tunedona
resentationstoexplicitlyencodelogicaloperators
newdataset(Hayesetal.,2019),evenifitsdistri-
withLearnedSparseRetrieval(LSR),showingthat
bution is similar to the original training data. In
penalizingnegationinthequeryimprovesgeneral-
certaincases,fine-tuningcanleadtoadegradation
ization.
ofperformanceintheoriginaltrainingset(Peters

3 RelatedWork 2020),oronlarger-scalenexttokenpredictionmod-
els,suchasLlama(Grattafiorietal.,2024),Mistral
NegationinIR.Negationhasbeenstudiedsince
(Jiang,2024)andQwen(Yangetal.,2024).
earlylanguagemodels,e.g.,JumeletandHupkes
Data generation using LLMs. Data generation
(2018)investigatethecapabilitiesofLSTMstolo-
usingLLMshasgainedsignificantattention(Abol-
cate the scope of negation, which they evaluate
ghasemi et al., 2024; Askari et al., 2023; Tun-
usingaparsetree. Earlyworktypicallyexamines
stall et al., 2023; Abbasiantaeb et al., 2024; Liu
negationattheatomicsentencelevel. Incontrast,
et al., 2024), and has been shown to be a viable
negation in IR must be handled across pairs of
methodtoexpandthetrainingdataset,improving
queriesanddocuments,asthepresenceofnegation
performanceinseveraltaskssuchasdialoggenera-
in a query can completely reverse the relevance
tion(Soudanietal.,2024b;Askarietal.,2025),rea-
ofadocumentthatotherwiseisasemanticmatch.
soning(Yinetal.,2023),negation(Lietal.,2023)
Therefore, IR systems must assess whether both
andexclusionaryretrieval(Zhangetal.,2024a).
thequeryandthedocumentsharethesamepolarity.
Existing negation datasets. One of the first for-
i.e., positive or negative (McQuire and Eastman,
aysintonegationunderstandingwasinthemedical
1998). NegationinIRoftentakestheformofex-
domain,whereresearchfocusedonautomatically
clusion,whichinvolvesfilteringinformation,and
indexingclinicalreportsanddischargesummaries
rejectionofsuggestions,whichinvolvesdismissing
(Savovaetal.,2010;Niuetal.,2005). Forexample,
information,asmentionedbyYaeger-DrorandTot-
Bio-Scope(Zhuetal.,2019)isacorpusofbiomed-
tie(1993). Havingdistincttypesofnegationposes
icaltextminingthatfocusesonextractingaccurate
anaddedchallengetodefiningitinanIRcontext,
informationonbiologicalrelations. Today,inthe
whichcanthereforebedifficultandambiguous.
IRliterature,wehaveaccesstopubliclyavailable
Negationindifferentmodalities. Alhamoudetal.
datasets such as NevIR (Weller et al., 2024), Ex-
(2025) propose a benchmark for understanding
cluIR(Zhangetal.,2024a),BoolQuestions(Zhang
negationacross18tasksandmodalitiesspanning
et al., 2024b), Quest (Malaviya et al., 2023), and
image,video,andmedicaldata. Theirexperiments
RomQA(Zhongetal.,2022). Whilethesedatasets
revealthatevenwithlarge-scaletraining,modern
containlogicaloperatorannotations,theannotation
visionlanguagemodels(VLMs)strugglewithnega-
systemlargelyremainsasinglebinarylabelforthe
tion,oftenperformingatrandom. Theauthorsshow
presenceofnegation.
that fine-tuning on large-scale synthetic datasets
Research gap. How is a taxonomy different
canapproacha10%increaseinperformance. How-
fromlinguisticformalisationsofnegationinlogic?
ever, that forces the model to overfit on negation
Aristotle transferred the study of negation from
insteadofmakingitreasononnegation,asshown
the domain of ontology to logic and language
by achieving a good performance on one dataset
(Smith, 2022). The linguistic formalization of
butnotgeneralizingonnegationoutofdistribution
negation in logic defines how negation operates
(Zhangetal.,2020;ZhouandSrikumar,2021).
within formal systems (da Costa, 1974), such as
Retrieval models and LLMs for retrieval. In-
inclassicallogic,whereapropositionpisnegated
formation retrieval models evolved from lexical
through¬pinwhichthetruthvalueisflipped,or
matching to dense retrieval, where the similarity
within modal and nonmonotonic logic (Ketsman
betweenaqueryanddocumentsisidentifiedina
and Koch, 2020), where it has more nuanced in-
latentsemanticspace. Theserepresentationscanbe
terpretations. Incontrast,ataxonomyfornegation
learnedseparately,i.e.,withbi-anddualencoders,
wouldcategorizedifferenttypesandfunctionsof
ortogether,i.e.,withcrossencoders. Densemod-
negationinlanguageandreasoning,suchaslexical
elshavebeenshowntooutperformclassicallexi-
(Staliunaite and Iacobacci, 2020) vs. semantical
calmatchinginmostscenarios(Karpukhinetal.,
(Urquhart, 1972) negation, metalinguistic (Horn,
2020; Khattab and Zaharia, 2020). In addition,
1985)vs. descriptive(Miestamo,2005;Lee,2017),
LLMsarebeingfine-tunedtoserveasthebackbone
ornegationasopposition(Mettinger,1994)vs. ab-
of retrieval and ranking tasks (Zhu et al., 2023),
sence(Faller,2002). Althoughlogictreatsnegation
bringingaboostinperformancethroughtheirrich
asaformaloperationontruthvalues,ataxonomy
representations. LLM-based models used for re-
exploresitsdiverserolesincommunication,cogni-
trievalareconstructedonsmall-scalemodels,such
tion,andinterpretation.
asBERT(Devlinetal.,2019)andT5(Raffeletal.,

4 Methodology
|            |     |            |     |     |          |         | words that    | are                                 | inherently                        |     | negative | through | their |
| ---------- | --- | ---------- | --- | --- | -------- | ------- | ------------- | ----------------------------------- | --------------------------------- | --- | -------- | ------- | ----- |
|            |     |            |     |     |          |         | meaning,e.g.: |                                     | refuse,deny,exclude,reject,avoid, |     |          |         |       |
| We propose | (1) | a taxonomy |     | for | negation | that is |               |                                     |                                   |     |          |         |       |
|            |     |            |     |     |          |         | lack,fail.    | Contrasting(Trillas,2017)negationis |                                   |     |          |         |       |
usedtogenerate(2)twosyntheticdatasetsthatcan
composedofwordsthatconveynegationinpairs,
| be used     | for evaluating |      | the    | performance |                 | of neural |                                   |     |          |       |      |            |           |
| ----------- | -------------- | ---- | ------ | ----------- | --------------- | --------- | --------------------------------- | --- | -------- | ----- | ---- | ---------- | --------- |
|             |                |      |        |             |                 |           | butarenotnegativeindependently.   |     |          |       |      | Thesecanbe |           |
| information | retrieval      |      | models | and         | for fine-tuning |           |                                   |     |          |       |      |            |           |
|             |                |      |        |             |                 |           | calledcontrastingpairsofantonyms. |     |          |       |      | Immediate  |           |
| models      | to become      | more | robust | on          | negation,       | and       |                                   |     |          |       |      |            |           |
|             |                |      |        |             |                 |           | antonyms                          | are | opposite | words | with | no         | degree of |
(3)aclassificationmechanismthatsplitsexisting
|     |     |     |     |     |     |     | variation | between | them; | Polar | antonyms |     | are op- |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------- | ----- | ----- | -------- | --- | ------- |
datasetsintogranulartypesofnegation.
|     |     |     |     |     |     |     | posite words | with | degrees |     | of variation |     | between |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ---- | ------- | --- | ------------ | --- | ------- |
them,andMidantonymsrepresentsamplesfrom
4.1 Taxonomy
|     |     |     |     |     |     |     | theinterpolationoftwopolarantonyms. |     |     |     |     |     | Formore |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | --- | ------- |
Wederiveournegationtaxonomyfromdefinitions
|     |     |     |     |     |     |     | special | cases of | negation | that | we  | do not | cover in |
| --- | --- | --- | --- | --- | --- | --- | ------- | -------- | -------- | ---- | --- | ------ | -------- |
inlogic,philosophy(Horn,1989)andnaturallan-
thisstudy,seeAppendixA.4.
guageprocessingliterature(Yaeger-DrorandTot-
| tie,1993;McQuireandEastman,1998). |     |     |     |     |     | Figure2 |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
4.2 DataGeneration
presentsthetaxonomyasahierarchicaltree,where
|           |         |     |          |      |     |           | We generate | two | synthetic |     | datasets | designed | to  |
| --------- | ------- | --- | -------- | ---- | --- | --------- | ----------- | --- | --------- | --- | -------- | -------- | --- |
| each node | denotes | a   | negation | type | and | its child |             |     |           |     |          |          |     |
coverallnegationtypesdescribedinthetaxonomy.
| nodescorrespondtofiner-grainedsubtypes. |     |     |     |     |     | Table |                                  |     |     |     |     |             |     |
| --------------------------------------- | --- | --- | --- | --- | --- | ----- | -------------------------------- | --- | --- | --- | --- | ----------- | --- |
|                                         |     |     |     |     |     |       | Weconstructthedatasetsasfollows: |     |     |     |     | (1)weprompt |     |
3inAppendixA.2includesquery-documentpairs
anLLMtogenerate100topicsofgeneralknowl-
exemplifyingeachnegationtype.
edgetoensurefamiliarity(Askarietal.,2025)and
| Our primary |     | classification |     | criterion |     | is on the |     |     |     |     |     |     |     |
| ----------- | --- | -------------- | --- | --------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
avoidlong-tailknowledge;(2)foreachtopic,we
| scope of | negation | (the | part | of a | sentence | whose |     |     |     |     |     |     |     |
| -------- | -------- | ---- | ---- | ---- | -------- | ----- | --- | --- | --- | --- | --- | --- | --- |
asktheLLMtoreturnoneWikipediapagethatwe
meaningisalteredbynegation),distinguishingex-
checkusingtheWikipediaAPI,ensuringthegen-
| plicit negation |        | realized | by      | a logical | operator | ¬       |          |              |     |               |     |     |         |
| --------------- | ------ | -------- | ------- | --------- | -------- | ------- | -------- | ------------ | --- | ------------- | --- | --- | ------- |
|                 |        |          |         |           |          |         | erations | are grounded |     | in documented |     | and | factual |
| (Haegeman,      | 1995), | from     | lexical |           | negation | that is |          |              |     |               |     |     |         |
information;(3)conditionedonaWikipediapage,
| present         | through | the                       | semantics | of       | the word | itself     |                        |       |          |       |         |        |             |
| --------------- | ------- | ------------------------- | --------- | -------- | -------- | ---------- | ---------------------- | ----- | -------- | ----- | ------- | ------ | ----------- |
|                 |         |                           |           |          |          |            | theLLMgeneratespairs(q |       |          |       | ,doc    | )and(q | ,doc )      |
|                 |         |                           |           |          |          |            |                        |       |          |       | 1       | 1      | 2 2         |
| (Natayou,2014). |         | LogicalOperatorsappendtoa |           |          |          |            |                        |       |          |       |         |        |             |
|                 |         |                           |           |          |          |            | following              | the   | template | of    | CondaQA |        | (Ravichan-  |
| word or         | clause, | reversing                 | its       | meaning. |          | In lexical |                        |       |          |       |         |        |             |
|                 |         |                           |           |          |          |            | der et al.,            | 2022) | and      | NevIR | (Weller | et     | al., 2024). |
negation,awordorphraseinherentlyevokesnega-
|     |     |     |     |     |     |     | (3.1) Given | detailed |     | prompts | constructed |     | for the |
| --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | --- | ------- | ----------- | --- | ------- |
tion,withouttheneedforanappendedoperator.
|             |     |       |       |            |     |           | individual | negation | type, | we  | ask | the LLM | to re- |
| ----------- | --- | ----- | ----- | ---------- | --- | --------- | ---------- | -------- | ----- | --- | --- | ------- | ------ |
| We identify |     | three | types | of logical |     | operators |            |          |       |     |     |         |        |
trieveaparagraphthatcontainsonespecificnega-
| based on | literature  |       | review   | (Horn, | 1989).       | Sen- |                |     |       |           |     |        |          |
| -------- | ----------- | ----- | -------- | ------ | ------------ | ---- | -------------- | --- | ----- | --------- | --- | ------ | -------- |
|          |             |       |          |        |              |      | tion asdefined |     | inthe | taxonomy. |     | If the | document |
| tential  | (Zeijlstra, | 2004) | negation |        | is signalled | by   |                |     |       |           |     |        |          |
doesnotcontainexplicitmarkersforthespecified
| sententialoperators |     |     | such as | no, | not and | none, |     |     |     |     |     |     |     |
| ------------------- | --- | --- | ------- | --- | ------- | ----- | --- | --- | --- | --- | --- | --- | --- |
negation,themodelwillretrievetheclosestmatch
| which have | a         | fixed  | syntactic | role      | and       | occupy |                                          |     |     |     |     |     |       |
| ---------- | --------- | ------ | --------- | --------- | --------- | ------ | ---------------------------------------- | --- | --- | --- | --- | --- | ----- |
|            |           |        |           |           |           |        | andrephraseitbyinjectingspecificmarkers, |     |     |     |     |     | i.e., |
| defined    | positions | within | a         | sentence. | Exclusion |        |                                          |     |     |     |     |     |       |
keywordssuchasimpossibleinsteadofnotpossi-
(MacCartneyandManning,2008)issignalledby
|                       |     |     |      |            |            |     | ble. This | phenomenon |     | was | observed | with | affixal |
| --------------------- | --- | --- | ---- | ---------- | ---------- | --- | --------- | ---------- | --- | --- | -------- | ---- | ------- |
| exclusionaryoperators |     |     | that | are either | exceptors, |     |           |            |     |     |          |      |         |
negations,whichourapproachtranslatedasasen-
suchasbesidesandothers(exceptorsrepresenta
|        |         |           |     |      |     |          | tential one,     | as  | they are                    | guaranteed |     | to be | semanti- |
| ------ | ------- | --------- | --- | ---- | --- | -------- | ---------------- | --- | --------------------------- | ---------- | --- | ----- | -------- |
| unique | type of | negation, | see | more | in  | Appendix |                  |     |                             |            |     |       |          |
|        |         |           |     |      |     |          | callyequivalent. |     | Theothertypesofnegationthat |            |     |       |          |
A.2),orquantifiers,suchastheuniversalquanti-
werenotalwayspresentinthedocumentswerethe
| fierforallandtheexistentialquantifierexists. |     |     |     |     |     | In  |     |     |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
quantifiers,whichcanbetranslatedfromonetothe
Aristotelianlogic(KeenanandWesterståhl,1997;
|                       |            |       |                |        |             |            | otherwithlogictransformations.        |            |     |            |           | (3.2)Giventhe |             |
| --------------------- | ---------- | ----- | -------------- | ------ | ----------- | ---------- | ------------------------------------- | ---------- | --- | ---------- | --------- | ------------- | ----------- |
| Horn, 1989),          |            | these | quantifiers    | define |             | three fun- |                                       |            |     |            |           |               |             |
|                       |            |       |                |        |             |            | extracted                             | paragraph, |     | the LLM    | generates |               | a query.    |
| damental              | relations: |       | Contradiction, |        | Contraries, |            |                                       |            |     |            |           |               |             |
|                       |            |       |                |        |             |            | Thisistheprocessofgeneratingonepair(q |            |     |            |           |               | 1 ,doc 1 ). |
| and Subcontradiction. |            |       | Finally,       |        | Affixal     | (Zim-      |                                       |            |     |            |           |               |             |
|                       |            |       |                |        |             |            | (3.3) For                             | generating |     | the second |           | pair, we      | employ      |
| mer, 1966)            | negation   |       | is signalled   |        | by prefix   | and        |                                       |            |     |            |           |               |             |
twostrategiestoproducedifferentdegreesoflexi-
| suffixoperators          |     | that | are preppended |                      | or  | appended |                                      |          |     |            |     |       |            |
| ------------------------ | --- | ---- | -------------- | -------------------- | --- | -------- | ------------------------------------ | -------- | --- | ---------- | --- | ----- | ---------- |
|                          |     |      |                |                      |     |          | caloverlapbetweenthenegateddatasets. |          |     |            |     |       | (1)Free    |
| toanexistingword,suchas: |     |      |                | un-,in-,im-,il-,ir-, |     |          |                                      |          |     |            |     |       |            |
|                          |     |      |                |                      |     |          | Generation:                          | generate |     | a positive |     | query | q 2 by re- |
dis-,non-,mis-,ill-,-less,-free(Wahyuni,2014).
|     |     |     |     |     |     |     | movingthenegationfromq |     |     |     | ; generateapositive |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | --- | ------------------- | --- | --- |
1
| We identify    |     | two types | of       | lexical | negation.   | Im- |             |                         |              |     |     |                |          |
| -------------- | --- | --------- | -------- | ------- | ----------- | --- | ----------- | ----------------------- | ------------ | --- | --- | -------------- | -------- |
|                |     |           |          |         |             |     | document    | doc                     | by answering |     | q . | (2) Controlled |          |
| plicit (Madva, |     | 2016)     | negation |         | is composed | of  |             | 2                       |              |     | 2   |                |          |
|                |     |           |          |         |             |     | Generation: | generateapositivequeryq |              |     |     | 2              | byremov- |

Sentential
(no,not,none)
|     |     |     |     |                  |     |     |           |     | Exceptors       |     | Contradiction       |          |
| --- | --- | --- | --- | ---------------- | --- | --- | --------- | --- | --------------- | --- | ------------------- | -------- |
|     |     |     |     |                  |     |     |           |     | (other,besides) |     | (forall..existsnot) |          |
|     |     |     |     | Logicaloperators |     |     | Exclusion |     |                 |     |                     |          |
|     |     |     |     |                  |     |     |           |     | Quantifiers     |     |                     | Contrary |
(forall..notexists)
Affixal
(in-,il-)
| Negationtaxonomy |     |     |     |     |     |     |     |     |     |     | Subcontradiction |     |
| ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- |
(some..somenot)
Immediateantonyms
Implicit
|     |     |     |     |     |     |     | (refuse,deny) |     | (north,south) |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ------------- | --- | --- | --- |
Lexical
Midantonyms
|     |     |     |     |     |     |     | Contrasting |     | (fast,moderate) |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --------------- | --- | --- | --- |
Polarantonyms
(fast,slow)
|     |     |     |     |     | Figure2: | Negationtaxonomytree. |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | -------- | --------------------- | --- | --- | --- | --- | --- | --- |
ingthenegationfromq 1 ;generateapositivedoc- icatesarefound,weanalysequeryanddocument
ument doc by removing the negation from doc . pairs. We extract the logical quantifiers present
|     | 2   |     |     |     |     | 1   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Thetwosyntheticallygenerateddatasetshave1505 in both the query and document (both pairs, see
and 1479 instances, respectively, where a single AppendixA.5),andcheckifanyofthefollowing
instance has pairs (q ,doc ) and (q ,doc ). Ap- logicalpatternsareidentifiedascontradiction,con-
|     |     |     | 1   | 1   | 2   | 2   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
pendix A.3 provides the prompts used for gener- traryandsubcontraditiondefinitions(Horn,1989):
ation,andanadditionalverificationstepforguar- (∀...∃¬),(∀...¬∃),(∃...∃¬). Instancesmatch-
anteeingtherelevanceofdocuments;Table4and inganyofthesepatternsarelabelledaccordingly.
Figure8summarizethedatasetstatisticsanddistri-
|     |     |     |     |     |     |     | Step3: | SemanticAntonymsDetection |     |     |     | We will |
| --- | --- | --- | --- | --- | --- | --- | ------ | ------------------------- | --- | --- | --- | ------- |
butionofgeneratedlabels.
|     |     |     |     |     |     |     | assume | the only | other potential |     | negation | is both |
| --- | --- | --- | --- | --- | --- | --- | ------ | -------- | --------------- | --- | -------- | ------- |
4.3 LMLogicclassification atthesemanticlevelandonlydetectableinpaired
interactions(incontrast,apredicatesuchasrefuse
| Negation        | can | be analysed                   |     | at two | granularities. |     |            |         |            |           |         |     |
| --------------- | --- | ----------------------------- | --- | ------ | -------------- | --- | ---------- | ------- | ---------- | --------- | ------- | --- |
|                 |     |                               |     |        |                |     | inherently | carries | a negative | polarity, | whereas | a   |
| Sentence-level: |     | somenegationtypescanbeidenti- |     |        |                |     |            |         |            |           |         |     |
fiedatthesentencelevel;iftwosentencesareeither predicate such as slow does not). We check such
antonympairswiththenltklibrary.
| both negative |     | or both | positive, | the | pair | agrees in |     |     |     |     |     |     |
| ------------- | --- | ------- | --------- | --- | ---- | --------- | --- | --- | --- | --- | --- | --- |
polarity(Mahanyetal.,2022),andiftheydonot,it
|     |     |     |     |     |     |     | Step4: | AbsenceofNegation |     | If  | none of | the pre- |
| --- | --- | --- | --- | --- | --- | --- | ------ | ----------------- | --- | --- | ------- | -------- |
conveysanegativepolarityrelationship(sentential,
viousconditionsaremet,weconcludethatthein-
| exclusionary,affixal,andimplicit). |     |     |     |     | Pair-level: | the |     |     |     |     |     |     |
| ---------------------------------- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
stancedoesnotcontainnegationaccordingtoour
| negation | polarity | can | only | be identified |     | by com- |     |     |     |     |     |     |
| -------- | -------- | --- | ---- | ------------- | --- | ------- | --- | --- | --- | --- | --- | --- |
taxonomy.
parison,i.e.,whetherbothstatementscanbetrue
atthesametime(quantifiersandcontrastingnega- 5 ExperimentalSetup
tion). Weproposeaclassificationmechanismthat
|         |               |     |       |          |         |        | Throughout | this | study, we | use the | GPT-4o-mini |     |
| ------- | ------------- | --- | ----- | -------- | ------- | ------ | ---------- | ---- | --------- | ------- | ----------- | --- |
| assigns | each instance |     | in an | existing | dataset | a cat- |            |      |           |         |             |     |
egory outlined in our taxonomy by converting it model (OpenAI et al., 2024) to conduct experi-
|            |       |       |       |        |     |          | ments that | aim | to answer | our research | questions. |     |
| ---------- | ----- | ----- | ----- | ------ | --- | -------- | ---------- | --- | --------- | ------------ | ---------- | --- |
| to natural | logic | using | typed | lambda | (λ) | calculus |            |     |           |              |            |     |
Moreprecisely,weevaluateretrievalmodelstore-
formalisations(Barendregt,1985)(seeAppendix
vealthenecessityofourtaxonomy-drivensynthetic
A.2). Wegenerateformalisationsforeachinstance
data,evaluatecategorizedexistingdatasetstoshow
bypromptingamodelwithaninstructiontogener-
theusefulnessofourlogic-drivenmechanism,and
atethetypedlambdacalculusproof,andreturnthe
fine-tunetoshowthatacoverageofnegationtypes
| predicates, | quantifiers |     | and | λ-typed | formula. | We  |     |     |     |     |     |     |
| ----------- | ----------- | --- | --- | ------- | -------- | --- | --- | --- | --- | --- | --- | --- |
categorizeanexistingdatasetinfouriterativesteps: canhelpwithgeneralisation.
|                    |                         |     |               |               |         |        | Evaluationofthegeneration. |           |          | Weassessthequal- |       |         |
| ------------------ | ----------------------- | --- | ------------- | ------------- | ------- | ------ | -------------------------- | --------- | -------- | ---------------- | ----- | ------- |
| Step1:             | PredicateClassification |     |               | Wecheckthere- |         |        |                            |           |          |                  |       |         |
|                    |                         |     |               |               |         |        | ity of the                 | generated | datasets | with             | human | annota- |
| turned predicates. |                         | If  | any predicate |               | defined | in the |                            |           |          |                  |       |         |
tionon5%ofthegenerations,withtwoannotators
deconstructionofthequeryisofsentential,exclu-
|     |     |     |     |     |     |     | evaluatingeachinstanceon: |     |     | (1)relevanceofdoc- |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------- | --- | --- | ------------------ | --- | --- |
sionary,affixal,orimplicitnature(asclassifiedby
|                                        |     |     |     |     |     |       | uments | to each | query, (2) | presence | of  | negation, |
| -------------------------------------- | --- | --- | --- | --- | --- | ----- | ------ | ------- | ---------- | -------- | --- | --------- |
| theLLM),welabeltheinstanceaccordingly. |     |     |     |     |     | Since |        |         |            |          |     |           |
(3)naturalness,(4)coherence,and(5)consistency
| they are    | sentence-level            |     | negations, |     | we only   | study |                        |           |            |               |     |        |
| ----------- | ------------------------- | --- | ---------- | --- | --------- | ----- | ---------------------- | --------- | ---------- | ------------- | --- | ------ |
|             |                           |     |            |     |           |       | of information         |           | within the | document.     | The | anno-  |
| thequeries. |                           |     |            |     |           |       | tation was             | conducted | with       | LabelStudio.4 |     | We as- |
| Step2:      | QuantifierPatternMatching |     |            |     | Ifnopred- |       | 4https://labelstud.io/ |           |            |               |     |        |

Model Architecture Trainingobjective Trainingdataset Size Tokenizer
| BM25            |     |     | Sparse     | Retrieval |     | N/A     |     | N/A  | N/A  |     |     |
| --------------- | --- | --- | ---------- | --------- | --- | ------- | --- | ---- | ---- | --- | --- |
| DPR[29]         |     |     | Bi-Encoder | Retrieval |     | NQ      |     | 219M | BERT |     |     |
| coCondenser[14] |     |     | Bi-Encoder | Retrieval |     | MSMarco |     | 110M | BERT |     |     |
| Dragon[37]      |     |     | Bi-Encoder | Retrieval |     | MSMARCO |     | N/A  | BERT |     |     |
msmarco-bert-base-dot-v5 DualEncoder SemanticSearch MSMarco 110M BERT
multi-qa-mpnet-base-dot-v1 DualEncoder SemanticSearch QA 110M MPNet
| Sentence-T5     |     |     | DualEncoder     | SentenceSimilarity |     | NLI     |     | 220M  | T5   |     |     |
| --------------- | --- | --- | --------------- | ------------------ | --- | ------- | --- | ----- | ---- | --- | --- |
| ColBERTv1[32]   |     |     | LateInteraction | Retrieval          |     | MSMarco |     | 110M  | BERT |     |     |
| ColBERTv2[59]   |     |     | LateInteraction | Retrieval          |     | MSMarco |     | 110M  | BERT |     |     |
| MonoT5Base[52]  |     |     | Crossencoder    | Ranking            |     | MSMarco |     | 223M  | T5   |     |     |
| MonoT5Large[52] |     |     | Crossencoder    | Ranking            |     | MSMarco |     | 737M  | T5   |     |     |
| MonoT53B[52]    |     |     | Crossencoder    | Ranking            |     | MSMarco |     | 2.85B | T5   |     |     |
stsb-roberta-large Crossencoder SentenceSimilarity STS-B 355M RoBERTa
| qnli-electra-base |     |     | Crossencoder | NLI |     | QNLI |     | 110M | ELECTRA |     |     |
| ----------------- | --- | --- | ------------ | --- | --- | ---- | --- | ---- | ------- | --- | --- |
nli-deberta-v3-base Crossencoder NLI MultiNLI,SNLI 184M DeBERTa
Qwen2-1.5B-Instruct[74] Transformer NTP Crawled 1.5B Qwen2Tokenizer
Qwen2-7B-Instruct[74] Transformer NTP Crawled 7B Qwen2Tokenizer
| Mistral-7B-Instruct[26]   |     |     | Transformer | NTP |     | Crawled |     | 7B  | BPE   |     |     |
| ------------------------- | --- | --- | ----------- | --- | --- | ------- | --- | --- | ----- | --- | --- |
| Llama-3.1-3B-Instruct[16] |     |     | Transformer | NTP |     | Crawled |     | 7B  | Llama |     |     |
| Llama-3.2-8B-Instruct[16] |     |     | Transformer | NTP |     | Crawled |     | 7B  | Llama |     |     |
Table1: Modelcomparisonforourexperiments. NLIreferstonaturallanguageinference,andNTPreferstonext
tokenprediction. bytepairencodingwithfallback. Thecrawleddatasetsrepresentundefinedlargetrainingsets.
sesstheannotationsonquantitativeandqualitative of construction: we generate data for each type
measures, togetherwiththeannotatoragreement. ofnegationconditionedonataxonomy-dependent
AppendixA.6illustratesthequestionsforthean- prompt. We run the classification mechanism on
notators,metricsused,alongsidefurtherdetailsfor thefreegenerationdataset,andobtainabalanced
the setup. For both performance and inner anno- accuracy score of 86.84% and an F1 score of
tatoragreement,weusemetricssuchasf1-score, 86.95%. We notice that around 54% of missclas-
averageonordinalscales,and(weighted)Cohen’s sificationsarecontrarynegationsmissclassifiedas
Kappa. Tables5and6reporttheannotationmet- contradictions. Inourexperiments,allmodelsper-
rics. Themainfindingsareasfollows: form similarly between these two types of nega-
• Annotatorsreported71–77%accuracyfordocu- tion,astheyarelogicallyandlexicallyverysimilar.
mentrelevanceand83%–88%f1scorefornega- Therefore,weassumeitdoesnotaffectourstudy.
| tionpresence. |     |     |     |     | Retrieval | Models. |     | We study | the performance |     | of  |
| ------------- | --- | --- | --- | --- | --------- | ------- | --- | -------- | --------------- | --- | --- |
• Onascaleof1–5,theannotatorsreportedanap- lexical,bi-encoder,cross-encoder,lateinteraction
proximatequalityof4onnaturalness,coherence,
|     |     |     |     |     | and | transformer | models | trained | for | first-stage | re- |
| --- | --- | --- | --- | --- | --- | ----------- | ------ | ------- | --- | ----------- | --- |
andconsistencyoflanguage. trieval, ranking, sentence similarity, natural lan-
• The inner annotator agreement passed signifi- guage inference (NLI) and next token prediction
cancevaluesforsententialandcontrastingnega- (NTP). We follow the experimental setup intro-
tion. Forimplicitandquantifiers,thetestshows
|     |     |     |     |     | ducedbyElsenetal.(2025). |     |     |     | Weshowthespecifi- |     |     |
| --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | ----------------- | --- | --- |
borderlineagreementinlanguagequality. cationsofallmodelsinTable1.
• Thebiggestdisagreementwasnoticedintheex- Datasets. We evaluate on three benchmarks.
ceptors.
|                     |     |        |           |          | NevIR | and | ExcluIR | are two | contrastive |     | bench- |
| ------------------- | --- | ------ | --------- | -------- | ----- | --- | ------- | ------- | ----------- | --- | ------ |
| • Human performance |     | on the | synthetic | datasets |       |     |         |         |             |     |        |
markswhereeachinstancecomprisesoftwodocu-
shows a pairwise accuracy score of 0.6571 ± mentsandtwoqueriesthatonlydifferbyatargeted
0.0202forfreegeneration,and(0.6643±0.0101
|     |     |     |     |     | negation, | or  | exclusion. | We  | also | use MSMarco |     |
| --- | --- | --- | --- | --- | --------- | --- | ---------- | --- | ---- | ----------- | --- |
forcontrolledgenerationonidentifyingtherele-
devpartition,whichisnotspecificallydesignedfor
vantdocumentforeachquestion. contrastivepairs,butisusedsimplyasacomplex
| Evaluationoftheclassificationmechanism. |     |     |     | We  | retrievalbenchmark. |     |     |     |     |     |     |
| --------------------------------------- | --- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- |
evaluatethequalityofourclassificationmechanism Metrics. The metric used to evaluate the task is
byassessingitagainstthegenerateddatasets,for
|               |        |           |        |           | pairwiseaccuracy: |           |        | foreachinstancequeriesq |     |               | ,q  |
| ------------- | ------ | --------- | ------ | --------- | ----------------- | --------- | ------ | ----------------------- | --- | ------------- | --- |
|               |        |           |        |           |                   |           |        |                         |     |               | 1 2 |
| which we have | access | to golden | labels | by design |                   |           |        |                         |     |               |     |
|               |        |           |        |           | and               | documents | d 1 ,d | 2 , the model           |     | independently |     |

ranks{d 1 ,d 2 }. Thepredictioniscorrectonlywhen existingdatasetshaveanunevenrepresentationon
the system places d above d for q and inverts negation,(H4)fine-tuningonoursyntheticallygen-
|     |     | 1   |     | 2   | 1   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
theorderforq . Randomperformanceforpairwise erateddatasetwillshowsystematicimprovement
2
| accuracyis25%. |     |              |     |       |         |      | inthedownstreamtaskpresentedinFigure1. |     |     |     |     |     |     |
| -------------- | --- | ------------ | --- | ----- | ------- | ---- | -------------------------------------- | --- | --- | --- | --- | --- | --- |
| Fine-tuning.   |     | We fine-tune |     | three | models: | Col- |                                        |     |     |     |     |     |     |
6.1 EvaluationonSyntheticData
BERTv1,multi-qa-mpnet-base-dot-v1,andMistral-
7B-Instruct for 20 epochs on the free generated Figure3illustrates20modelsevaluatedonthefree
|     |     |     |     |     |     |     | generation | dataset. |     | Sparse, | dual, | and biencoders |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | --- | ------- | ----- | -------------- | --- |
datasetandevaluateonNevIR(Welleretal.,2024)
exhibitpoorperformanceonalltypesofnegation,
testandMSMarco(Bajajetal.,2016)devdata.
|     |     |     |     |     |     |     | except   | Sentence-T5: |     | a dual | encoder          | trained | for     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------------ | --- | ------ | ---------------- | ------- | ------- |
|     |     |     |     |     |     |     | semantic | similarity.  |     | Both   | late-interaction |         | and all |
cross-encodermodels,exceptnli-deberta-v3-base,
|     |     |     |     |     |     |     | show strong | performance |     |     | on all | negation | types. |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ----------- | --- | --- | ------ | -------- | ------ |
BERTandT5-basedcross-encodersperformbetter
thanmodelswithaRoBERTa,ELECTRA,andDe-
|     |     |     |     |     |     |     | BERTabackbone. |     | Alltransformer-basedmodels, |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --------------------------- | --- | --- | --- | --- |
exceptforQwen1.5B(whichhasadisadvantagein
size,andwhichhasbeentrainedforNTP)perform
wellonalmostallnegationtypes.
Weperformaone-wayANOVAtotestthesig-
|     |     |     |     |     |     |     | nificance | of the | results. | ON  | model | architecture, |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------ | -------- | --- | ----- | ------------- | --- |
theANOVAtestreportsap-valueof1.0087e−11,
andtheTukeyHSDshowsasignificantdifference
|     |     |     |     |     |     |     | between     | sparse       | and   | dense      | models.         | When        | group-    |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------------ | ----- | ---------- | --------------- | ----------- | --------- |
|     |     |     |     |     |     |     | ing on      | the training |       | objective, | ANOVA           |             | indicates |
|     |     |     |     |     |     |     | p = 1.5709e |              | − 04, | with       | significant     | differences |           |
|     |     |     |     |     |     |     | between     | combinations |       | of         | NTP, retrieval, |             | and se-   |
manticsearch,andbetweensentencesimilarityvs.
|     |     |     |     |     |     |     | retrieval.  | Thetestshowsastatisticallysignificant |     |           |     |           |       |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ------------------------------------- | --- | --------- | --- | --------- | ----- |
|     |     |     |     |     |     |     | difference  | between                               |     | exceptors | and | all other | types |
|     |     |     |     |     |     |     | ofnegation. | Theexperimentsconfirmhypothesis       |     |           |     |           |       |
Figure 3: Pairwise Accuracy on the free generations H1andH2,thatis,somenegationtypesarebetter
encodedthanothers,andthatmodelspecifics,such
| dataset. | The | first result | column |     | contains | the full |     |     |     |     |     |     |     |
| -------- | --- | ------------ | ------ | --- | -------- | -------- | --- | --- | --- | --- | --- | --- | --- |
dataset; later columns represent one negation type asarchitectureandtrainingobjective,influenceper-
| each. Models |     | are represented |     | by  | the rows, | where |           |                                     |     |     |     |     |     |
| ------------ | --- | --------------- | --- | --- | --------- | ----- | --------- | ----------------------------------- | --- | --- | --- | --- | --- |
|              |     |                 |     |     |           |       | formance. | Ananalysisonthecontrolledgeneration |     |     |     |     |     |
I is a shortcut for Instruct. On the right, we as- datasetisillustratedinFigure11inAppendixA.7,
signlabelsexpressingthearchitectureandtrainingob-
whereasimilarbehaviorisseen;however,thepat-
| jective of | each | model: | the | first position |     | shows the |     |     |     |     |     |     |     |
| ---------- | ---- | ------ | --- | -------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
ternsareevenstronger,withageneraltrendtoward
| architecture, | i.e., | Sparse,          | Bi-encoder, |     | Dual   | encoder, |        |              |     |      |        |          |        |
| ------------- | ----- | ---------------- | ----------- | --- | ------ | -------- | ------ | ------------ | --- | ---- | ------ | -------- | ------ |
|               |       |                  |             |     |        |          | higher | performance. |     | This | can be | inherent | in the |
| Crossencoder, |       | and Transformer; |             | the | second | position |        |              |     |      |        |          |        |
datagenerationprocess,i.e.,document2isgener-
| shows the | training | objective, |     | i.e., | Retrieval, | Search, |     |     |     |     |     |     |     |
| --------- | -------- | ---------- | --- | ----- | ---------- | ------- | --- | --- | --- | --- | --- | --- | --- |
Similarity,Ranking,NaturalLanguageInference,and ated by changing the negation in document 1 (as
| NextTokenPrediction. |     |     | Foraclose-up,seeAppendix |     |     |     |     |     |     |     |     |     |     |
| -------------------- | --- | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
comparedtodirectlyansweringquery2).
A.7.
6.2 EvaluationonLogicFilteredNevIR
6 Results
|     |     |     |     |     |     |     | When we | apply | the | classification |     | mechanism | on  |
| --- | --- | --- | --- | --- | --- | --- | ------- | ----- | --- | -------------- | --- | --------- | --- |
thevalidationsetofNevIR,wefindthatthreemain
Ourexperimentsaredesignedtoinvestigatethefol-
|                   |     |                          |     |     |     |     | types of | negation | are        | present. | Out          | of  | 225 pairs, |
| ----------------- | --- | ------------------------ | --- | --- | --- | --- | -------- | -------- | ---------- | -------- | ------------ | --- | ---------- |
| lowinghypotheses: |     | (H1)somenegationtypesare |     |     |     |     |          |          |            |          |              |     |            |
|                   |     |                          |     |     |     |     | {79, 54, | 44}      | correspond | to       | {Sentential, |     | Affixal,   |
betterencodedinthemodelinternalrepresentations
Implicit},while31havebeenclassifiedasnotcon-
thanothers,(H2)modelspecificssuchasarchitec-
tainingnegation,inwhichcasewelabelasOthers,
ture,trainingobjective,sizeandbackbonesignif-
whiletheremaining17pairsarespreadacrossthe
| icantly influence |     | performance |     | on  | negation, | (H3) |             |     |          |         |     |               |     |
| ----------------- | --- | ----------- | --- | --- | --------- | ---- | ----------- | --- | -------- | ------- | --- | ------------- | --- |
|                   |     |             |     |     |           |      | other types | of  | negation | present | in  | the taxonomy. |     |

|     |     |     |     |     | among | the other | classes. |     | This means | that | more |
| --- | --- | --- | --- | --- | ----- | --------- | -------- | --- | ---------- | ---- | ---- |
than81%oftheentiredatasethasbeenclassified
|     |     |     |     |     | asexclusionary. |     | Theseresultsfurthersupporthy- |     |     |     |     |
| --- | --- | --- | --- | --- | --------------- | --- | ----------------------------- | --- | --- | --- | --- |
pothesisH3.
AsshowninFigure12(AppendixA.7),theper-
|     |     |     |     |     | formance                                  | of themodel |              | is  | approximately |      | uniform  |
| --- | --- | --- | --- | --- | ----------------------------------------- | ----------- | ------------ | --- | ------------- | ---- | -------- |
|     |     |     |     |     | betweenthethreeidentifiedtypesofnegation. |             |              |     |               |      | This     |
|     |     |     |     |     | finding                                   | contradicts | with         | our | synthetic     | data | exper-   |
|     |     |     |     |     | iments,                                   | where       | exclusionary |     | negation      | was  | signifi- |
cantlymoredifficulttoencodethantheothertypes
|     |     |     |     |     | of negation. | To  | further | inspect | the | source | of this |
| --- | --- | --- | --- | --- | ------------ | --- | ------- | ------- | --- | ------ | ------- |
discrepancy,wetakeacloserinspectionoftheEx-
|     |     |     |     |     | cluIR instances |     | identified |     | as “Sentential” |     | or “Im- |
| --- | --- | --- | --- | --- | --------------- | --- | ---------- | --- | --------------- | --- | ------- |
plicit”. Thisrevealsthattheseinstancesonlyhavea
differentrephrasingofataskthatessentiallyisstill
|     |     |     |     |     | exclusion. | Oneexampleextractedfromthedataset |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---------- | --------------------------------- | --- | --- | --- | --- | --- |
is‘CanyoutellmeaboutPaulZiert’sinvolvement
infoundingtheBartConnerGymnasticsAcademy
inNorman,Oklahoma,whileavoidinganymention
|     |     |     |     |     | of Bart     | Conner’s  | role | in the     | academy?’. |               | Our cat- |
| --- | --- | --- | --- | --- | ----------- | --------- | ---- | ---------- | ---------- | ------------- | -------- |
|     |     |     |     |     | egorization | mechanism |      | identifies |            | this instance | as       |
“Implicit”,whileithastheformofasetsubtraction,
asperthedefinitionofexceptors.
6.4 Fine-tuning
Wefine-tuneColBERTv1,multiqa-mpnet-base-dot-
v1,andMistral-7B-Instructonthefreegeneration
|     |     |     |     |     | dataset,  | NevIR,                              | and | a mixed | strategy | with | both |
| --- | --- | --- | --- | --- | --------- | ----------------------------------- | --- | ------- | -------- | ---- | ---- |
|     |     |     |     |     | datasets. | Weevaluatethefinetunedmodelsagainst |     |         |          |      |      |
Figure4: PairwiseAccuracyonNevIRassplitwithour
NevIRdevsetandMSMarcodevsmall.
classificationmechanism.
|     |     |     |     |     | Trainpartitions: |     | TheNevIRtrainingsetiscom- |     |     |     |     |
| --- | --- | --- | --- | --- | ---------------- | --- | ------------------------- | --- | --- | --- | --- |
TheseresultsareinlinewithhypothesisH3,which
|     |     |     |     |     | posedof1,896triplets. |     |     | Thetrainpartitionofour |     |     |     |
| --- | --- | --- | --- | --- | --------------------- | --- | --- | ---------------------- | --- | --- | --- |
statesthatexistingdatasetshaveanunevendistri-
|     |     |     |     |     | synthetically |     | generated | dataset | consists |     | of 2,114 |
| --- | --- | --- | --- | --- | ------------- | --- | --------- | ------- | -------- | --- | -------- |
butionofnegationtypes.
triplets. Whenfine-tuningmixeddata,wehavea
Figure4showsthatmodelsperformworseonthe
totalof2,005triplets.
NevIRdatasetcomparedtooursyntheticallygener-
|                                |                                   |     |           |     | Evaluation     | partitions: |          | We      | evaluate | against       | the |
| ------------------------------ | --------------------------------- | --- | --------- | --- | -------------- | ----------- | -------- | ------- | -------- | ------------- | --- |
| ateddataset.                   | Sentence-T5exhibitsthebestperfor- |     |           |     |                |             |          |         |          |               |     |
|                                |                                   |     |           |     | test partition |             | of NevIR | that    | has      | 2.8k triplets | (2  |
| manceamongbi-anddual-encoders. |                                   |     | ColBERTv1 |     |                |             |          |         |          |               |     |
|                                |                                   |     |           |     | triplets       | = 1 pair),  | and      | against | the dev  | partition     | of  |
hasahigherperformancethanColBERTv2,andthe
MSMarco.
| MonoT5    | models perform                     | the best | on all | types of |     |     |     |     |     |     |     |
| --------- | ---------------------------------- | -------- | ------ | -------- | --- | --- | --- | --- | --- | --- | --- |
| negation. | SimilarlytoFigure3,wenoticethatthe |          |        |          |     |     |     |     |     |     |     |
6.4.1 EvaluationonNevIR
performanceinallmodelsforsententialnegation
AsshowninTable2andinFigure13inAppendix
| ishigherthanaffixalorimplicit. |     | Qwen2-1.5Bper- |     |     |     |     |     |     |     |     |     |
| ------------------------------ | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
A.7.1,fine-tuningColBERTandMultiQAonour
formstheworstofallLLMs,similarlytosynthetic
syntheticdatasetyieldsanimmediateperformance
experiments.
gainontheNevIRdevelopmentset,howeverpeak-
ingwhilefine-tuningonNevIRtrainreacheshigher
6.3 EvaluationonLogicFilteredExcluIR
|     |     |     |     |     | performance |     | in the | last epoch. | This | is  | to be ex- |
| --- | --- | --- | --- | --- | ----------- | --- | ------ | ----------- | ---- | --- | --------- |
Whenapplyingtheclassificationmechanismonthe
pectedasforthesyntheticdataweevaluateOOD.
| ExcluIR      | test set, we find | three types | of   | negation: |           |                 |     |     |              |     |       |
| ------------ | ----------------- | ----------- | ---- | --------- | --------- | --------------- | --- | --- | ------------ | --- | ----- |
|              |                   |             |      |           | To assess | in-distribution |     |     | performance, | we  | apply |
| {Sentential, | Exclusionary,     | Implicit}   | with | {189,     |           |                 |     |     |              |     |       |
mixedfine-tuningbycombiningthetwodatasets
| 2820, 113}pairsoutof3452. |     | Moreover,297have |     |     |               |     |           |     |       |          |      |
| ------------------------- | --- | ---------------- | --- | --- | ------------- | --- | --------- | --- | ----- | -------- | ---- |
|                           |     |                  |     |     | and shuffling |     | the data. | The | model | achieves | high |
beenclassifiedas“Other”while32aredistributed
performancesignificantlyfasterthanwhensimply

NevIRP.Acc.↑ MSMarcoMRR@10↑
E1 E6 E20 E1 E6 E20
TREBloC NevIR .21 .24 .45 .37 .37 .34
Synth .23 .33 .36 .36 .34 .31
Mixed .23 .40 .48 .37 .33 .31
AQitluM NevIR .12 .51 .52 .35 .17 .06
Synth .34 .38 .40 .33 .07 .03
Mixed .36 .52 .50 .26 .03 .01
lartsiM
negationtypes,and(3)fine-tuningonoursynthetic
datasetshelpsperformanceinanegationdomain.
These insights confirm that negation is a com-
plex phenomenon and that a thorough taxonomy
brings advantages as a starting point for generat-
ing fine-tuning data. The taxonomy-based classi-
fication of current datasets, together with model
evaluation, shows that having a broad coverage
ofnegationtypesisvital. Ourfine-tuningexperi-
NevIR .70 .78 .78 .53 .58 .60 ments confirm that the synthetic datasets bring a
Synth .58 .58 .58 .59 .55 .55 performanceboost;however,italsoindicatesthat
Mixed .72 .78 .78 .57 .60 .54
fine-tuningdatamightnotbethesolefactorbehind
Table 2: Results for ColBERT, MultiQA and Mistral model difficulty with negation. The training ob-
when trained on NevIR, Synth and Mixed data, and
jectiveandarchitecturalbackboneplayabigrole
evaluatedonNevIRandMSMarco. ColumnsE0,E1,
inmodelperformanceperformance. However,dif-
E6, E20 represent epochs 0 (before backprop.), 1, 6
ferenttrainingobjectivesareapromisingdirection
and 20; P. Acc. stands for pairwise accuracy, while
for future work. Moreover, we propose investi-
MRR@10formeanreciprocalrankat10.
gating negation in a retrieval setting with a large
fine-tuned on NevIR, giving the overall best per-
corpora. Moreover,whilegeneralizationdropswith
formance. Mistralshowsthesamebehaviourwith
fine-tuning,weproposeinvestigatingthetraining
mixed fine-tuning. This supports hypothesis H4,
objective by applying reinforcement learning on
thatoursyntheticallygenerateddatasethelpsincap-
negationwithasmallsubset,similartoR1-Search
turingnegation. Overall,wenoticethatfine-tuning
(Jinetal.,2025).
onoursyntheticdatabringsaquickperformance
boostagainsttheNevIRdevandtestsets,indicat-
ingthatourproposeddatasetscapturethenotionof
negation.
6.4.2 EvaluationonMSMarco
When evaluated against MSMarco (Table 2 and
Figure14inAppendixA.7.1), wenoticethatthe
generalizabilityofColBERTandMultiQAdrops
whenfine-tunedonanydataset. Interestingly,Mis-
tral displays a more stable fine-tuning process;
however,addingsyntheticdatadropsperformance
evenfurther. AlthoughMSMarcogeneralizationis
knowntobenegativelyaffectedwhenmodelsare
fine-tuned out of distribution, our results show a
trade-off: syntheticandmixedtraininghelpsgen-
eralisation in the negation domain, but it further
harmsgeneralisationonMSMarco.
7 Conclusion
Inthisstudy, weproposeaphilosophy, logicand
linguistic-groundedtaxonomyfornegationalong
two synthetic datasets that can be used for evalu-
ating existing neural retrieval, ranking and LLM
reranker models, and for fine-tuning models to
increase their capabilities on negation. Through
our study, we found that (1) cross-encoders and
LLMrerankersarebetteratencodingnegation,(2)
NevIR and ExcluIR have a limited coverage of

Limitations
|                                            |     |           |             |     |     |         | Marzyeh           | Ghassemi.            |     | 2025.     | Vision-language |            | mod-     |
| ------------------------------------------ | --- | --------- | ----------- | --- | --- | ------- | ----------------- | -------------------- | --- | --------- | --------------- | ---------- | -------- |
|                                            |     |           |             |     |     |         | els do            | not understand       |     | negation. |                 | arXiv      | preprint |
| Ourworkproposesanewdatasetforinvestigating |     |           |             |     |     |         | arXiv:2501.09425. |                      |     |           |                 |            |          |
| negation                                   | and | improving | performance |     | in  | a nega- |                   |                      |     |           |                 |            |          |
|                                            |     |           |             |     |     |         | ArianAskari,      | MohammadAliannejadi, |     |           |                 | ChuanMeng, |          |
tionsetting,andafilteringmechanismforstudying
|          |           |          |     |           |         |      | EvangelosKanoulas,andSuzanVerberne.2023. |     |     |                       |     |     | Ex- |
| -------- | --------- | -------- | --- | --------- | ------- | ---- | ---------------------------------------- | --- | --- | --------------------- | --- | --- | --- |
| existing | datasets. | However, |     | there are | certain | lim- |                                          |     |     |                       |     |     |     |
|          |           |          |     |           |         |      | pand,highlight,generate:                 |     |     | RL-drivendocumentgen- |     |     |     |
itations to our study. Our dataset is limited to a erationforpassagereranking. InProceedingsofthe
binaryclassificationredefinedasapairwiserank- 2023ConferenceonEmpiricalMethodsinNatural
LanguageProcessing,pages10087–10099.
ingtask,andthereforeisnotdirectlyapplicableto
a ranking setting with a large corpus. Moreover, Arian Askari, Roxana Petcu, Chuan Meng, Moham-
|                                    |     |     |     |     |          |     | mad Aliannejadi,                |     | Amin | Abolghasemi, |     | Evangelos    |     |
| ---------------------------------- | --- | --- | --- | --- | -------- | --- | ------------------------------- | --- | ---- | ------------ | --- | ------------ | --- |
| thedataisgeneratedusingGPT-4omini. |     |     |     |     | Whilethe |     |                                 |     |      |              |     |              |     |
|                                    |     |     |     |     |          |     | Kanoulas,andSuzanVerberne.2025. |     |      |              |     | Self-seeding |     |
faithfulnessofinformationisnotthedirectscope
andmulti-intentself-instructingLLMsforgenerat-
ofthispaper,havingamorecontrolledgeneration
|     |     |     |     |     |     |     | ingintent-awareinformation-seekingdialogs. |     |     |     |     |     | arXiv |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | ----- |
processwouldbebeneficial. Lastly,abroaderstudy preprintarXiv:2402.11633.
| on datasets | such | as BoolQuestions, |     |     | RomQA | and |                     |     |                               |     |     |     |     |
| ----------- | ---- | ----------------- | --- | --- | ----- | --- | ------------------- | --- | ----------------------------- | --- | --- | --- | --- |
|             |      |                   |     |     |       |     | JayDavidAtlas.1977. |     | Negation,ambiguity,andpresup- |     |     |     |     |
Questwouldofferamoreextensivestudy.
|                 |     |     |     |     |     |     | position.                                    | LinguisticsandPhilosophy,1(3):321–336. |     |     |     |     |     |
| --------------- | --- | --- | --- | --- | --- | --- | -------------------------------------------- | -------------------------------------- | --- | --- | --- | --- | --- |
| Acknowledgments |     |     |     |     |     |     | PayalBajaj,DanielCampos,NickCraswell,LiDeng, |                                        |     |     |     |     |     |
JianfengGao,XiaodongLiu,RanganMajumder,An-
| The evaluation |     | of our | generated | data | was | done |                |     |         |     |        |             |     |
| -------------- | --- | ------ | --------- | ---- | --- | ---- | -------------- | --- | ------- | --- | ------ | ----------- | --- |
|                |     |        |           |      |     |      | drew McNamara, |     | Bhaskar |     | Mitra, | Tri Nguyen, | Mir |
throughLabelStudio. Moreover,weacknowledge Rosenberg,XiaSong,AlinaStoica,SaurabhTiwary,
ourcolleagueswhohelpedwithhumanevaluation andTongWang.2016. MSMARCO:Ahumangener-
|                 |     |            |     |               |     |        | atedmachinereadingcomprehensiondataset. |     |     |     |     |     | arXiv |
| --------------- | --- | ---------- | --- | ------------- | --- | ------ | --------------------------------------- | --- | --- | --- | --- | --- | ----- |
| and annotation: |     | Panagiotis |     | Eustratiadis, |     | Jasmin |                                         |     |     |     |     |     |       |
preprintarXiv:1611.09268.
| Kareem, | ClaraRus,     |     | DavidVos, | MariaHeuss, |      | Lu     |                        |     |     |                    |     |     |     |
| ------- | ------------- | --- | --------- | ----------- | ---- | ------ | ---------------------- | --- | --- | ------------------ | --- | --- | --- |
|         |               |     |           |             |      |        | HenkP.Barendregt.1985. |     |     | TheLambdaCalculus: |     |     | Its |
| Zhang,  | and Catherine |     | Chen.     | We also     | want | to ac- |                        |     |     |                    |     |     |     |
knowledgeMariaAloni,whoofferedhelpandfeed- SyntaxandSemantics. North-Holland.
backforourlinguisticstudy.
|     |     |     |     |     |     |     | NewtonC.A.daCosta.1974. |     |     |     | Onthetheoryofincon- |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | --- | ------------------- | --- | --- |
This research was (partially) supported by the sistentformalsystems. NotreDameJ.FormalLog.,
| Dutch Research |              | Council | (NWO),           |     | under | project | 15:497–510.   |          |     |        |        |     |          |
| -------------- | ------------ | ------- | ---------------- | --- | ----- | ------- | ------------- | -------- | --- | ------ | ------ | --- | -------- |
| numbers        | 024.004.022, |         | NWA.1389.20.183, |     |       | and     |               |          |     |        |        |     |          |
|                |              |         |                  |     |       |         | Jacob Devlin, | Ming-Wei |     | Chang, | Kenton |     | Lee, and |
KICH3.LTP.20.006, the European Union under Kristina Toutanova. 2019. BERT: Pre-training of
deepbidirectionaltransformersforlanguageunder-
| grantagreementsNo. |     |          | 101070212(FINDHR)and |           |           |     |              |     |             |         |        |                 |     |
| ------------------ | --- | -------- | -------------------- | --------- | --------- | --- | ------------ | --- | ----------- | ------- | ------ | --------------- | --- |
|                    |     |          |                      |           |           |     | standing.    | In  | Proceedings |         | of the | 2019 conference |     |
| No. 101201510      |     | (UNITE), |                      | and Ahold | Delhaize. |     |              |     |             |         |        |                 |     |
|                    |     |          |                      |           |           |     | of the North |     | American    | chapter | of     | the association |     |
Viewsandopinionsexpressedarethoseoftheau-
|     |     |     |     |     |     |     | forcomputationallinguistics: |     |     |     | humanlanguagetech- |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | --- | ------------------ | --- | --- |
thor(s)onlyanddonotnecessarilyreflectthoseof nologies, volume 1 (long and short papers), pages
| theirrespectiveemployers,fundersand/orgranting |     |     |     |     |     |     | 4171–4186. |     |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
authorities.
|     |     |     |     |     |     |     | TommasoDolci.2022.                      |     |     | Fine-tuninglanguagemodels |     |     |      |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | --- | ------------------------- | --- | --- | ---- |
|     |     |     |     |     |     |     | tomitigategenderbiasinsentenceencoders. |     |     |                           |     |     | 2022 |
IEEEEighthInternationalConferenceonBigData
References
|     |     |     |     |     |     |     | Computing | Service |     | and Applications |     | (BigDataSer- |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------- | --- | ---------------- | --- | ------------ | --- |
vice),pages175–176.
ZahraAbbasiantaeb,YifeiYuan,EvangelosKanoulas,
| andMohammadAliannejadi.2024. |     |     |     |     | LettheLLMs |     |          |            |          |     |          |         |      |
| ---------------------------- | --- | --- | --- | --- | ---------- | --- | -------- | ---------- | -------- | --- | -------- | ------- | ---- |
|                              |     |     |     |     |            |     | Coen van | den Elsen, | Francien |     | Barkhof, | Thijmen | Nij- |
talk:Simulatinghuman-to-humanconversationalQA
|     |     |     |     |     |     |     | dam, Simon | Lupart, |     | and Mohammad |     | Alliannejadi. |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ------- | --- | ------------ | --- | ------------- | --- |
viazero-shotLLM-to-LLMinteractions. InProceed- 2025. ReproducingNevIR:Negationinneuralinfor-
ingsofthe17thACMInternationalConferenceon
|     |     |     |     |     |     |     | mationretrieval. |     | arXivpreprintarXiv:2502.13506. |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ------------------------------ | --- | --- | --- | --- |
WebSearchandDataMining,pages8–17.
|     |     |     |     |     |     |     | Martina Faller. | 2002. |     | Semantics | and | Pragmatics | of  |
| --- | --- | --- | --- | --- | --- | --- | --------------- | ----- | --- | --------- | --- | ---------- | --- |
AminAbolghasemi,ZhaochunRen,ArianAskari,Mo-
|     |     |     |     |     |     |     | EvidentialsinCuzcoQuechua. |     |     |     | Ph.D.thesis,Stanford |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | --- | -------------------- | --- | --- |
hammadAliannejadi,MaartendeRijke,andSuzan
University.
| Verberne.2024. |     | Cause: | Counterfactualassessment |     |     |     |     |     |     |     |     |     |     |
| -------------- | --- | ------ | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
of user satisfaction estimation in task-oriented dia- LuyuGaoandJamieCallan.2021. Unsupervisedcor-
loguesystems. arXivpreprintarXiv:2403.19056. pusawarelanguagemodelpre-trainingfordensepas-
|        |           |        |             |            |          |     | sageretrieval. |     | arXivpreprintarXiv:2108.05540. |     |     |     |     |
| ------ | --------- | ------ | ----------- | ---------- | -------- | --- | -------------- | --- | ------------------------------ | --- | --- | --- | --- |
| Kumail | Alhamoud, | Shaden | Alshammari, |            | Yonglong |     |                |     |                                |     |     |     |     |
| Tian,  | Guohao    | Li,    | Philip      | Torr, Yoon | Kim,     | and |                |     |                                |     |     |     |     |

EmmaJ.Gerritse,FaeghehHasibi,andArjenP.deVries. Jaap Jumelet and Dieuwke Hupkes. 2018. Do
2022. Entity-awaretransformersforentitysearch. In language models understand anything? On the
SIGIR’22: The45thInternationalACMSIGIRCon- ability of LSTMs to understand negative polarity
ferenceonResearchandDevelopmentinInformation items. InProceedingsoftheWorkshop: Analyzing
Retrieval,Madrid,Spain,July11-15,2022,pages and Interpreting Neural Networks for NLP, Black-
| 1455–1465.ACM. |     |     |     |     |     | boxNLP@EMNLP2018,Brussels,Belgium,Novem- |     |     |     |     |
| -------------- | --- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- |
ber1,2018,pages222–231.AssociationforCompu-
| AaronGrattafiori,AbhimanyuDubey,AbhinavJauhri, |                  |     |         |       |     | tationalLinguistics. |     |     |     |     |
| ---------------------------------------------- | ---------------- | --- | ------- | ----- | --- | -------------------- | --- | --- | --- | --- |
| Abhinav                                        | Pandey, Abhishek |     | Kadian, | Ahmad | Al- |                      |     |     |     |     |
Dahle,AieshaLetman,AkhilMathur,AlanSchelten, Vladimir Karpukhin, Barlas Oguz, Sewon Min,
AlexVaughan,and1others.2024. TheLlama3herd PatrickSHLewis,LedellWu,SergeyEdunov,Danqi
arXivpreprintarXiv:2407.21783.
| ofmodels.             |     |                          |     |     |     | Chen, and   | Wen-tau     | Yih. 2020. | Dense passage | re- |
| --------------------- | --- | ------------------------ | --- | --- | --- | ----------- | ----------- | ---------- | ------------- | --- |
|                       |     |                          |     |     |     | trieval for | open-domain | question   | answering.    | In  |
| LilianeHaegeman.1995. |     | TheSyntaxofNegation,vol- |     |     |     |             |             |            |               |     |
EMNLP,pages6769–6781.
| ume75. | CambridgeUniversityPress. |     |     |     |     |                                       |     |     |          |     |
| ------ | ------------------------- | --- | --- | --- | --- | ------------------------------------- | --- | --- | -------- | --- |
|        |                           |     |     |     |     | EdwardL.KeenanandDagWesterståhl.1997. |     |     | General- |     |
JosephYHalpernandJudeaPearl.2005. Causesand izedquantifiersinlinguisticsandlogic. InHandbook
explanations: A structural-model approach. Part I: ofLogicandLanguage.
| Causes. | The British | Journal | for the | Philosophy | of  |     |     |     |     |     |
| ------- | ----------- | ------- | ------- | ---------- | --- | --- | --- | --- | --- | --- |
Science,56:843–887. Bas Ketsman and Christoph E. Koch. 2020. Datalog
|     |     |     |     |     |     | with negation | and | monotonicity. | In International |     |
| --- | --- | --- | --- | --- | --- | ------------- | --- | ------------- | ---------------- | --- |
TylerL.Hayes,KushalKafle,RobikShrestha,Manoj
ConferenceonDatabaseTheory.
| Acharya,andChristopherKanan.2019. |     |     |     | REMIND |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
yourneuralnetworktopreventcatastrophicforget- OmarKhattabandMateiZaharia.2020. Colbert: Effi-
ting. arXivpreprintarXiv:1910.02509. cientandeffectivepassagesearchviacontextualized
|     |     |     |     |     |     | lateinteractionoverbert. |     | InProceedingsofthe43rd |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------ | --- | ---------------------- | --- | --- |
LaurenceR.Horn.1985. Metalinguisticnegationand InternationalACMSIGIRConferenceonResearch
pragmaticambiguity. Language,61:121–174. and Development in Information Retrieval, pages
39–48.
| LaurenceR.Horn.1989. |     | ANaturalHistoryofNegation. |     |     |     |     |     |     |     |     |
| -------------------- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
UniversityofChicagoPress.
AntoniosMinasKrasakis,AndrewYates,andEvangelos
|                      |     |                            |     |     |     | Kanoulas.2025.                               |     | Constructingset-compositionaland |     |       |
| -------------------- | --- | -------------------------- | --- | --- | --- | -------------------------------------------- | --- | -------------------------------- | --- | ----- |
| LaurenceR.Horn.2010. |     | MultiplenegationinEnglish  |     |     |     |                                              |     |                                  |     |       |
|                      |     |                            |     |     |     | negatedrepresentationsforfirst-stageranking. |     |                                  |     | CoRR, |
| andotherlanguages.   |     | InTheExpressionofNegation, |     |     |     |                                              |     |                                  |     |       |
abs/2501.07679.
pages111–148.DeGruyterMoutonBerlin,Boston.
|     |     |     |     |     |     | KennethKunen.1987. |     | Negationinlogicprogramming. |     |     |
| --- | --- | --- | --- | --- | --- | ------------------ | --- | --------------------------- | --- | --- |
Md Mosharaf Hossain, Venelin Kovatchev, Pranoy TheJournalofLogicProgramming,4(4):289–308.
| Dutta, Tiffany | Kao, | Elizabeth | Wei, | and Eduardo |     |     |     |     |     |     |
| -------------- | ---- | --------- | ---- | ----------- | --- | --- | --- | --- | --- | --- |
Blanco.2020. Ananalysisofnaturallanguageinfer- ChungminLee.2017. Metalinguisticnegationvs.de-
ence benchmarks through the lens of negation. In scriptivenegation: Amongtheirkinandfoes. InThe
| Proceedings | of the 2020 | Conference |     | on Empirical |     |            |              |          |           |      |
| ----------- | ----------- | ---------- | --- | ------------ | --- | ---------- | ------------ | -------- | --------- | ---- |
|             |             |            |     |              |     | Pragmatics | of Negation: | Negative | meanings, | uses |
MethodsinNaturalLanguageProcessing(EMNLP).
anddiscursivefunctions.JohnBenjaminsPublishing
Company.
ArianHosseini,SivaReddy,DzmitryBahdanau,RDe-
vonHjelm,AlessandroSordoni,andAaronCourville. JudithYueLi,ArenJansen,QingqingHuang,Joonseok
2021. Understanding by understanding not: Mod- Lee,RaviGanti,andDimaKuzmin.2023. MAQA:
elingnegationinlanguage models. arXivpreprint A multimodal QA benchmark for negation. arXiv
arXiv:2105.03519.
preprintarXiv:2301.03238.
XiaoshuiHuang,ShengLi,WentaoQu,TongHe,Yifan
Sheng-ChiehLin,AkariAsai,MinghanLi,BarlasOguz,
| Zuo,andWanliOuyang.2022. |     |     | FrozenCLIPmodel |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- |
JimmyLin,YasharMehdad,Wen-tauYih,andXilun
isanefficientpointcloudbackbone. arXivpreprint Chen. 2023. How to train your dragon: Diverse
arXiv:2212.04098. augmentationtowardsgeneralizabledenseretrieval.
arXivpreprintarXiv:2302.07452.
| FengqingJiang.2024. | Identifyingandmitigatingvul- |     |     |     |     |     |     |     |     |     |
| ------------------- | ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
nerabilitiesinLLM-integratedapplications. Master’s ZiyiLin, ShijieGeng, RenruiZhang, PengGao, Ger-
thesis,UniversityofWashington.
|            |             |         |      |         |       | arddeMelo,              | XiaogangWang, |                        | JifengDai, Y.Qiao,  |     |
| ---------- | ----------- | ------- | ---- | ------- | ----- | ----------------------- | ------------- | ---------------------- | ------------------- | --- |
|            |             |         |      |         |       | andHongshengLi.2022.    |               |                        | FrozenCLIPmodelsare |     |
| Bowen Jin, | Hansi Zeng, | Zhenrui | Yue, | Jinsung | Yoon, |                         |               |                        |                     |     |
|            |             |         |      |         |       | efficientvideolearners. |               | InEuropeanConferenceon |                     |     |
SercanArik,DongWang,HamedZamani,andJiawei
ComputerVision.
| Han.2025. | Search-R1:TrainingLLMstoreasonand |     |     |     |     |     |     |     |     |     |
| --------- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
leveragesearchengineswithreinforcementlearning. ZihanLiu,WeiPing,RajarshiRoy,PengXu,Chankyu
arXivpreprintarXiv:2503.09516. Lee, Mohammad Shoeybi, and Bryan Catanzaro.
|     |     |     |     |     |     | 2024. ChatQA:BuildingGPT-4levelconversational |                                      |     |     |     |
| --- | --- | --- | --- | --- | --- | --------------------------------------------- | ------------------------------------ | --- | --- | --- |
|     |     |     |     |     |     | QAmodels.                                     | arXivpreprintarXiv:arXiv:2401.10225. |     |     |     |

Bill MacCartney and Christopher D. Manning. 2008. EMNLP2020,pages708–718,Online.Association
Modelingsemanticcontainmentandexclusioninnat- forComputationalLinguistics.
| urallanguageinference. |     |     | InInternationalConference |     |     |     |     |     |     |     |     |
| ---------------------- | --- | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
onComputationalLinguistics. Hiroshi Noji and Hiroya Takamura. 2020. An anal-
|     |     |     |     |     |     | ysis of | the utility | of explicit | negative | examples | to  |
| --- | --- | --- | --- | --- | --- | ------- | ----------- | ----------- | -------- | -------- | --- |
AlexMadva.2016. Whyimplicitattitudesare(proba- improve the syntactic abilities of neural language
bly)notbeliefs. Synthese,193:2659–2684. models. arXivpreprintarXiv:2004.02451.
AhmedMahany,HebaKhaled,NouhSabriElmitwally, OpenAI,AaronHurst,AdamLerer,AdamP.Goucher,
NaifAljohani,andSaidGhoniemy.2022. Negation Adam Perelman, Aditya Ramesh, Aidan Clark,
andspeculationinNLP:Asurvey,corpora,methods, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec
andapplications. AppliedSciences,12(10):5209. Radford,AleksanderMa˛dry,AlexBaker-Whitcomb,
|     |     |     |     |     |     | Alex Beutel, |     | Alex Borzunov, |     | Alex Carney, | Alex |
| --- | --- | --- | --- | --- | --- | ------------ | --- | -------------- | --- | ------------ | ---- |
Chaitanya Malaviya, Peter Shaw, Ming-Wei Chang, Chow, Alex Kirillov, Alex Nichol, and 400 oth-
| KentonLee,andKristinaToutanova.2023. |         |     |                   |     | QUEST:  |            |        |        |       |       |          |
| ------------------------------------ | ------- | --- | ----------------- | --- | ------- | ---------- | ------ | ------ | ----- | ----- | -------- |
|                                      |         |     |                   |     |         | ers. 2024. | GPT-4o | system | card. | arXiv | preprint |
| A retrieval                          | dataset |     | of entity-seeking |     | queries |            |        |        |       |       |          |
arXiv:2410.21276.
| with implicit | set | operations. |     | arXiv | preprint |     |     |     |     |     |     |
| ------------- | --- | ----------- | --- | ----- | -------- | --- | --- | --- | --- | --- | --- |
arXiv:2305.11694. LourdesOrtega,AndreaTyler,HaeInPark,andMariko
|     |     |     |     |     |     | Uno. 2016. | The | Usage-based |     | Study of Language |     |
| --- | --- | --- | --- | --- | --- | ---------- | --- | ----------- | --- | ----------------- | --- |
Ian R. McKenzie, Alexander Lyzhov, Michael Pieler, LearningandMultilingualism. GeorgetownUniver-
| AliciaParrish,AaronMueller,AmeyaPrabhu,Euan |     |     |     |     |     | sityPress. |     |     |     |     |     |
| ------------------------------------------- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- |
McLean,AaronKirtland,AlexisRoss,AlisaLiu,An-
drewGritsevskiy,DanielWurgaft,DerikKauffman, MatthewE.Peters,SebastianRuder,andNoahA.Smith.
GabrielRecchia,JiachengLiu,JoeCavanagh,Max 2019. Totuneornottotune? Adaptingpretrained
Weiss,SicongHuang,TheFloatingDroid,and8oth- representations to diverse tasks. arXiv preprint
| ers.2024. | Inversescaling: |     | Whenbiggerisn’tbetter. |     |     | arXiv:1903.05987. |     |     |     |     |     |
| --------- | --------------- | --- | ---------------------- | --- | --- | ----------------- | --- | --- | --- | --- | --- |
arXivpreprintarXiv:2306.09479.
ColinRaffel,NoamShazeer,AdamRoberts,Katherine
AprilR.McQuireandCarolineM.Eastman.1998. The Lee,SharanNarang,MichaelMatena,YanqiZhou,
ambiguityofnegationinnaturallanguagequeriesto Wei Li, and Peter J Liu. 2020. Exploring the lim-
informationretrievalsystems. J.Am.Soc.Inf.Sci., its of transfer learning with a unified text-to-text
| 49:686–692. |     |     |     |     |     | transformer. | JournalofMachineLearningResearch, |     |     |     |     |
| ----------- | --- | --- | --- | --- | --- | ------------ | --------------------------------- | --- | --- | --- | --- |
21(140):1–67.
| Amil Merchant, | Elahe | Rahimtoroghi, |     | Ellie | Pavlick, |     |     |     |     |     |     |
| -------------- | ----- | ------------- | --- | ----- | -------- | --- | --- | --- | --- | --- | --- |
and Ian Tenney. 2020. What happens to BERT AbhilashaRavichander,MattGardner,andAnaMaraso-
embeddings during fine-tuning? arXiv preprint vic´.2022. CONDAQA:Acontrastivereadingcom-
arXiv:2004.14448. prehension dataset for reasoning about negation.
arXivpreprintarXiv:2211.00295.
| ArthurMettinger.1994. |     | AspectsofSemanticOpposition |     |     |     |     |     |     |     |     |     |
| --------------------- | --- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
inEnglish. OxfordUniversityPress. Keshav Santhanam, Omar Khattab, Jon Saad-Falcon,
|                     |     |                   |     |     |          | Christopher | Potts,    | and Matei | Zaharia.  | 2021.     | Col- |
| ------------------- | --- | ----------------- | --- | --- | -------- | ----------- | --------- | --------- | --------- | --------- | ---- |
| MattiMiestamo.2005. |     | StandardNegation: |     |     | TheNega- |             |           |           |           |           |      |
|                     |     |                   |     |     |          | bertv2:     | Effective | and       | efficient | retrieval | via  |
tionofDeclarativeVerbalMainClausesinaTypo- lightweight late interaction. arXiv preprint
| logicalPerspective. |     | DeGruyterMouton. |     |     |     | arXiv:2112.01488. |     |     |     |     |     |
| ------------------- | --- | ---------------- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- |
Roser Morante and Walter Daelemans. 2012. Guergana K. Savova, James J. Masanz, Philip V.
ConanDoyle-neg: Annotationofnegationcuesand Ogren,JiapingZheng,SunghwanSohn,KarinKip-
| theirscopeinConanDoylestories. |     |     |     | InProceedings |     |                                        |     |     |     |     |      |
| ------------------------------ | --- | --- | --- | ------------- | --- | -------------------------------------- | --- | --- | --- | --- | ---- |
|                                |     |     |     |               |     | perSchuler,andChristopherG.Chute.2010. |     |     |     |     | Mayo |
oftheEighthInternationalConferenceonLanguage
clinicaltextanalysisandknowledgeextractionsys-
Resources and Evaluation, LREC 2012, Istanbul, tem(ctakes): architecture,componentevaluationand
Turkey, May 23-25, 2012, pages 1563–1568. applications. JournaloftheAmericanMedicalInfor-
EuropeanLanguageResourcesAssociation(ELRA). maticsAssociation,175:507–13.
RosyaneFlorineNatayou.2014. ExplicitandImplicit JulianJSchlöderandRaquelFernández.2015. Prag-
| MeansofNegationintheEnglishLanguage. |     |     |     |     | Ph.D. |                  |     |                |     |             |        |
| ------------------------------------ | --- | --- | --- | --- | ----- | ---------------- | --- | -------------- | --- | ----------- | ------ |
|                                      |     |     |     |     |       | matic rejection. |     | In Proceedings |     | of the 11th | Inter- |
thesis,SumyStateUniversity.
|     |     |     |     |     |     | national | Conference | on Computational |     | Semantics, |     |
| --- | --- | --- | --- | --- | --- | -------- | ---------- | ---------------- | --- | ---------- | --- |
pages250–260.
YunNiu,Xiao-DanZhu,JianhuaLi,andGraemeHirst.
2005. Analysis of polarity information in medi- GeorgeOSeiver.1944. Cicero’sdeoratoreandrabelais.
caltext. InAMIAAnnualSymposiumProceedings, PMLA,59(3):655–671.
pages570–574.
|     |     |     |     |     |     | Robin Smith. | 2022. | Aristotle’s | Logic. | In Edward | N.  |
| --- | --- | --- | --- | --- | --- | ------------ | ----- | ----------- | ------ | --------- | --- |
RodrigoNogueira,ZhiyingJiang,RonakPradeep,and ZaltaandUriNodelman,editors,TheStanfordEncy-
Jimmy Lin. 2020. Document ranking with a pre- clopediaofPhilosophy,Winter2022edition.Meta-
| trained sequence-to-sequence |     |     | model. |     | In Findings |     |     |     |     |     |     |
| ---------------------------- | --- | --- | ------ | --- | ----------- | --- | --- | --- | --- | --- | --- |
physicsResearchLab,StanfordUniversity.
| of the Association |     | for | Computational |     | Linguistics: |     |     |     |     |     |     |
| ------------------ | --- | --- | ------------- | --- | ------------ | --- | --- | --- | --- | --- | --- |

HeydarSoudani,EvangelosKanoulas,andFaeghehHa- WenhaoZhang,MengqiZhang,ShiguangWu,Jiahuan
sibi.2024a. Finetuningvs.retrievalaugmentedgen- Pei, Zhaochun Ren, Maarten de Rijke, Zhumin
erationforlesspopularknowledge. InProceedings Chen, and Pengjie Ren. 2024a. ExcluIR: Exclu-
ofthe2024AnnualInternationalACMSIGIRCon- sionaryneuralinformationretrieval. arXivpreprint
| ferenceonResearchandDevelopmentinInformation |     |     |     | arXiv:2404.17288. |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- |
RetrievalintheAsiaPacificRegion,SIGIR-AP2024,
Tokyo, Japan, December 9-12, 2024, pages 12–22. ZongmengZhang,JinhuaZhu,WengangZhou,Xiang
| ACM. |     |     |     | Qi,PengZhang,andHouqiangLi.2024b. |     |     |     |     | BoolQues- |
| ---- | --- | --- | --- | --------------------------------- | --- | --- | --- | --- | --------- |
tions: Doesdenseretrievalunderstandbooleanlogic
HeydarSoudani, RoxanaPetcu, EvangelosKanoulas, inlanguage? InConferenceonEmpiricalMethods
and Faegheh Hasibi. 2024b. A survey on recent inNaturalLanguageProcessing.
arXiv
| advances | in conversational | data generation. |     |     |     |     |     |     |     |
| -------- | ----------------- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
VictorZhong,WeijiaShi,WentauYih,andLukeZettle-
preprintarXiv:2405.13003.
|     |     |     |     | moyer. | 2022. | RoMQA: | A benchmark |     | for robust, |
| --- | --- | --- | --- | ------ | ----- | ------ | ----------- | --- | ----------- |
IevaStaliunaiteandIgnacioIacobacci.2020. Composi- multi-evidence, multi-answer question answering.
tionalandlexicalsemanticsinRoBERTa,BERTand arXivpreprintarXiv:2210.14353.
| DistilBERT:AcasestudyonCoQA. |     | arXivpreprint |     |     |     |     |     |     |     |
| ---------------------------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
arXiv:2009.08257. YichuZhouandVivekSrikumar.2021. Acloserlook
|                     |                               |                      |     | at how            | fine-tuning | changes | BERT.  | arXiv | preprint |
| ------------------- | ----------------------------- | -------------------- | --- | ----------------- | ----------- | ------- | ------ | ----- | -------- |
| EnricTrillas.2017.  | Antonyms.negation,andthefuzzy |                      |     | arXiv:2106.14282. |             |         |        |       |          |
| case. InOntheLogos: |                               | ANaïveViewonOrdinary |     |                   |             |         |        |       |          |
|                     |                               |                      |     | Yanjie Zhu,       | Yuanyuan    | Liu,    | Leslie | Ying, | Xin Liu, |
ReasoningandFuzzyLogic,pages25–34.
|     |     |     |     | HairongZheng,andDongLiang.2019. |     |     |     |     | Bio-scope: |
| --- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- | ---------- |
Lewis Tunstall, Edward Beeching, Nathan Lambert, fast biexponential T1ρ mapping of the brain us-
Nazneen Rajani, Kashif Rasul, Younes Belkada, ingsignal-compensatedlow-rankplussparsematrix
Shengyi Huang, Leandro Von Werra, Clémentine decomposition. Magnetic Resonance in Medicine,
| Fourrier,NathanHabib,and1others.2023. |     |               | Zephyr: | 83:2092–2106. |     |     |     |     |     |
| ------------------------------------- | --- | ------------- | ------- | ------------- | --- | --- | --- | --- | --- |
| DirectdistillationofLMalignment.      |     | arXivpreprint |         |               |     |     |     |     |     |
YutaoZhu,HuayingYuan,ShutingWang,JiongnanLiu,
arXiv:2310.16944.
WenhanLiu,ChenlongDeng,HaonanChen,Zheng
AlasdairUrquhart.1972. Semanticsforrelevantlogics. Liu,ZhichengDou,andJi-RongWen.2023. Large
JournalofSymbolicLogic,37:159–169. languagemodelsforinformationretrieval: Asurvey.
arXivpreprintarXiv:2308.07107.
| SriWahyuni.2014. | Ananalysisonaffixalnegationin |     |     |     |     |     |     |     |     |
| ---------------- | ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
English. S1Thesis.UniversityofMataram. KarlE.Zimmer.1966. Affixalnegationinenglishand
|     |     |     |     | otherlanguages: |     | Aninvestigationofrestrictedpro- |     |     |     |
| --- | --- | --- | --- | --------------- | --- | ------------------------------- | --- | --- | --- |
Yuxia Wang, Minghan Wang, Muhammad Arslan ductivity. Language,42:134.
| Manzoor, | Fei Liu, Georgi | Georgiev, | Rocktim Jy- |     |     |     |     |     |     |
| -------- | --------------- | --------- | ----------- | --- | --- | --- | --- | --- | --- |
oti Das, and Preslav Nakov. 2024. Factuality of AriannaZuanazzi,PabloRipollés,WyMingLin,Laura
large language models: A survey. arXiv preprint Gwilliams, Jean-Rémi King, and David Poeppel.
arxiv:2402.02420. 2023. Tracking the behavioral and neural dynam-
|                                             |          |                       |          | ics of   | semantic | representations |     | through | negation. |
| ------------------------------------------- | -------- | --------------------- | -------- | -------- | -------- | --------------- | --- | ------- | --------- |
| OrionWeller,DawnLawrie,andBenjaminVanDurme. |          |                       |          | bioRxiv. |          |                 |     |         |           |
| 2024. NevIR:                                | Negation | in neural information | re-      |          |          |                 |     |         |           |
| trieval. arXivpreprintarXiv:2305.07614.     |          |                       |          |          |          |                 |     |         |           |
| MalcahYaeger-DrorandGunnelTottie.1993.      |          |                       | Negation |          |          |                 |     |         |           |
| inenglishspeechandwriting:                  |          | Astudyinvariation.    |          |          |          |                 |     |         |           |
Language,69:590.
AnYang,BaosongYang,BeichenZhang,BinyuanHui,
BoZheng,BowenYu,ChengyuanLi,DayihengLiu,
| FeiHuang,HaoranWei,and1others.2024. |                                |     | Qwen2. |     |     |     |     |     |     |
| ----------------------------------- | ------------------------------ | --- | ------ | --- | --- | --- | --- | --- | --- |
| 5technicalreport.                   | arXivpreprintarXiv:2412.15115. |     |        |     |     |     |     |     |     |
XunjianYin,BaizhouHuang,andXiaojunWan.2023.
ALCUNA:Largelanguagemodelsmeetnewknowl-
edge. Preprint,arXiv:2310.14820.
| HeddeZeijlstra.2004. | SententialNegationandNega- |     |     |     |     |     |     |     |     |
| -------------------- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
tiveConcord.
Ph.D.thesis,LOT.
TianyiZhang,FelixWu,ArzooKatiyar,KilianQWein-
| berger,andYoavArtzi.2020. |                                | Revisitingfew-sample |     |     |     |     |     |     |     |
| ------------------------- | ------------------------------ | -------------------- | --- | --- | --- | --- | --- | --- | --- |
| BERTfine-tuning.          | arXivpreprintarXiv:2006.05987. |                      |     |     |     |     |     |     |     |

A Appendix
ceptorsareinherentlyadifferenttypeofnegation
|     |     |     |     |     |     |     | comparedtotherestofthetaxonomy. |     |     |     |     | Thisdiffer- |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- | ----------- | --- |
Thisappendixoffersfurthermaterialthatsupports
encemightinfluencehowmodelsperformonthis
| thestudy.  | Itisorganisedasfollows: |            |             |     | AppendixA.1 |         |                 |       |                            |      |              |     |       |
| ---------- | ----------------------- | ---------- | ----------- | --- | ----------- | ------- | --------------- | ----- | -------------------------- | ---- | ------------ | --- | ----- |
|            |                         |            |             |     |             |         | negation        | type. | We also                    | give | a definition | of  | typed |
| defines    | the properties          |            | of negation |     | that are    | briefly |                 |       |                            |      |              |     |       |
|            |                         |            |             |     |             |         | lambdacalculus. |       | Moreover,weprovideexamples |      |              |     |       |
| referenced | in                      | the study. | Appendix    |     | A.2 gives   | an      |                 |       |                            |      |              |     |       |
foreachnegationtypepresentinthetaxonomyin
exampleinaninformationretrievalstyleforeach
themoviedomaintoexemplifythenegationtypes
typeofnegationpresentinthetaxonomy,alongside
|     |     |     |     |     |     |     | inaretrievalsetting. |     | Theexamplesareillustrated |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------- | --- | ------------------------- | --- | --- | --- | --- |
furtherdefinitionsofexceptorsandtypedlambda
inTable3.
| calculus. | AppendixA.3listsallthepromptsused |     |     |     |     |     |     |     |     |     |     |     |     |
| --------- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Exceptorsrepresentauniquetypeofnegation.
| to generate | the | datasets. | Appendix |     | A.4 mentions |     |           |       |          |       |      |          |     |
| ----------- | --- | --------- | -------- | --- | ------------ | --- | --------- | ----- | -------- | ----- | ---- | -------- | --- |
|             |     |           |          |     |              |     | While the | other | negation | types | take | the form | of  |
usecasesthatwedonotexplicitlyaccountforin
opposition,i.e.,twopropositionspand¬pcannot
| this study,                   | although |      | they are | interesting | to             | study. |                   |        |                           |            |     |     |        |
| ----------------------------- | -------- | ---- | -------- | ----------- | -------------- | ------ | ----------------- | ------ | ------------------------- | ---------- | --- | --- | ------ |
|                               |          |      |          |             |                |        | be true           | at the | same time,                | exceptions |     | are | a form |
| A.5 lists                     | details  | into | applying | the         | categorization |        |                   |        |                           |            |     |     |        |
|                               |          |      |          |             |                |        | ofsetsubtraction. |        | Moreprecisely,ifwedenotea |            |     |     |        |
| mechanismontheExcluIRdataset. |          |      |          |             | AppendixA.6    |        |                   |        |                           |            |     |     |        |
domainS={allcandidateanswers},anexception
includesthesurveythatthehumanannotatorscom-
|                |            |                               |             |            |                |          | setE ⊆                             | S = {itemstoexclude}andanexclusion- |       |         |                  |            |     |
| -------------- | ---------- | ----------------------------- | ----------- | ---------- | -------------- | -------- | ---------------------------------- | ----------------------------------- | ----- | ------- | ---------------- | ---------- | --- |
| pleted         | to perform | a                             | qualitative | evaluation |                | of the   |                                    |                                     |       |         |                  |            |     |
|                |            |                               |             |            |                |          | ary query                          | Q                                   | = S \ | E, then | any              | document   | D   |
| generateddata. |            | AppendixA.7containstheresults |             |            |                |          |                                    | ex                                  |       |         |                  |            |     |
|                |            |                               |             |            |                |          | thatsatisfiestheexclusionaryqueryQ |                                     |       |         |                  | willinher- |     |
| of evaluating  |            | the models                    |             | against    | the controlled |          |                                    |                                     |       |         |                  | ex         |     |
|                |            |                               |             |            |                |          | entlysatisfythewholesetS           |                                     |       |         | asaconsequenceof |            |     |
| generated      | dataset    | and                           | the         | ExcluIR    | data.          | Finally, |                                    |                                     |       |         |                  |            |     |
|                |            |                               |             |            |                |          | S \E                               | ⊆ S.                                |       |         |                  |            |     |
AppendixA.6.1offersastatisticalanalysisofthe
Typedlambdacalculusisaformalsystemthat
annotator’sanswers.
|     |     |     |     |     |     |     | decomposes | any | statement |     | into a | logic form, | by  |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --------- | --- | ------ | ----------- | --- |
definingabstractpredicatesanddeterminers,either
A.1 NegationProperties
assumingtheirtruthvalue,orreachingunitclauses
DrawinginspirationfromMoranteandDaelemans
|     |     |     |     |     |     |     | that can | only | be True | or only | False | (reaching | a   |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---- | ------- | ------- | ----- | --------- | --- |
(2012),wedefinethefollowingpropertiesofnega-
|     |     |     |     |     |     |     | contradiction). |     | Theprimarygoaloftypedlambda |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | --------------------------- | --- | --- | --- | --- |
tion:
|     |     |     |     |     |     |     | calculus | is to | provide | a framework |     | for meaning |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----- | ------- | ----------- | --- | ----------- | --- |
• Negationcues: Negationcuescanbesingle compositionwithflexiblefunctions(predicatesand
| words,    | multiwords, |         | prefixes, |      | such as   | im-, or | determiners).      |     |     |     |     |     |     |
| --------- | ----------- | ------- | --------- | ---- | --------- | ------- | ------------------ | --- | --- | --- | --- | --- | --- |
| suffixes, |             | such as | -less.    | They | introduce | the     |                    |     |     |     |     |     |     |
|           |             |         |           |      |           |         | A.3 DataGeneration |     |     |     |     |     |     |
negationinthesentence.
She did not go to the movies, but Inthissection,weshowthepromptsusedforgener-
Example:
atingthesyntheticdatasetsforfreeandcontrolled
wenttothetheaterinstead.
|     |     |     |     |     |     |     | generation. | Weillustratethepromptforgenerating |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ---------------------------------- | --- | --- | --- | --- | --- |
• Negated event: The main event or property sentential negation in Figure 5. The prompts for
thatisbeingnegated. Forexample,ifwede- generatingexceptors,affixalandimplicitnegation
fine¬asanegationoperation,i.e. ¬A,then aresimilar,whereonlysteps1and2aredifferent.
Aisthenegatedevent. Weillustratesteps1and2foreachofthesenega-
|          |     |         |     |       |             |     | tiontypesinFigure7. |     |     | Thepromptsforcontrasting |     |     |     |
| -------- | --- | ------- | --- | ----- | ----------- | --- | ------------------- | --- | --- | ------------------------ | --- | --- | --- |
| Example: |     | She did | not | go to | the movies, | but |                     |     |     |                          |     |     |     |
clausesandquantifiersareshowninFigure6.
wenttothetheaterinstead.
|           |     |        |           |     |                |     | Extra Verification |     | for | the | generated | instances. |     |
| --------- | --- | ------ | --------- | --- | -------------- | --- | ------------------ | --- | --- | --- | --------- | ---------- | --- |
| • Negated |     | scope: | Extension |     | of the negated |     |                    |     |     |     |           |            |     |
Aftergeneration,wefiltertheinstancesbyprompt-
event;partofthesentencewherethenegation
|     |     |     |     |     |     |     | ing the | LLM to | check | the relevance |     | of the | docu- |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------ | ----- | ------------- | --- | ------ | ----- |
propagates and changes its semantics. The mentsforthequeries. Weonlykeeptheinstances
partsofthesentencethatarenotaffectedby
forwhichbothpairspasstherelevanceself-check.
negationshouldbeleftoutthescope.
|                          |     |         |     |                |     |     | Thisverification                  |         | stepisneededas               |              |     | sometimesthe |         |
| ------------------------ | --- | ------- | --- | -------------- | --- | --- | --------------------------------- | ------- | ---------------------------- | ------------ | --- | ------------ | ------- |
|                          |     |         |     |                |     |     | generated                         | queries | are                          | too general, |     | making       | the re- |
| Example:                 |     | She did | not | gotothemovies, |     | but |                                   |         |                              |              |     |              |         |
| wenttothetheaterinstead. |     |         |     |                |     |     | trieveddocumentnothighlyrelevant. |         |                              |              |     |              |         |
|                          |     |         |     |                |     |     | LabelDistribution.                |         | Figure8illustratesthedistri- |              |     |              |         |
A.2 Taxonomy
butionofnegationtypespersyntheticdatasetafter
| In this | section, | we give | a definition |     | of exceptors |     |           |              |     |       |           |      |        |
| ------- | -------- | ------- | ------------ | --- | ------------ | --- | --------- | ------------ | --- | ----- | --------- | ---- | ------ |
|         |          |         |              |     |              |     | the extra | verification |     | step. | We notice | that | out of |
usingsetoperations,supportingourclaimthatex-

| epocS Negation | Negation    | Aristotelian |                                   |       |
| -------------- | ----------- | ------------ | --------------------------------- | ----- |
|                |             |              | Examples                          | Level |
| category       | subcategory | logic        |                                   |       |
| Sentential     |             |              | Q:MoviesthatdonotfeatureTomHanks. |       |
Sentence
| (no,not,none) |     |     | D:ForrestGumpfeaturesTomHanks. |     |
| ------------- | --- | --- | ------------------------------ | --- |
Exceptors
Q:MovieswithTomHanksbesidesForrestGump.
|                  | (others,besides |     |                                       | Sentence |
| ---------------- | --------------- | --- | ------------------------------------- | -------- |
| srotarepolacigoL |                 |     | D:ForrestGumpisawidelyacclaimedmovie. |          |
but,except)
Q:WhatareallmovieswithTomHanks?
|           |     | Contradiction |                                      | Pair |
| --------- | --- | ------------- | ------------------------------------ | ---- |
| Exclusion |     |               | D:HerearesomemovieswithoutTomHanks.. |      |
Q:WhatareallmovieswithTomHanks?
|     | Quantifiers | Contrary |     | Pair |
| --- | ----------- | -------- | --- | ---- |
D:ThereexistnomovieswithTomHanks.
Q:WhataresomemovieswithTomHanks?
|     |     | Subcontradiction |     | Pair |
| --- | --- | ---------------- | --- | ---- |
D:HerearesomemovieswithoutTomHanks.
Q:Whataresomemovieswithunhappyendings?
| Affixal |     |     |     | Sentence |
| ------- | --- | --- | --- | -------- |
D:Thesemovieshavehappyendings.
Q:ArethereanymovieswithTomHanks
| Implicit |     |     | thatfailedpeople’sexpectations?. | Sentence |
| -------- | --- | --- | -------------------------------- | -------- |
D:Thismoviesucceededinpublic’seye.
lacixeL
Q:Amoviethatisprofessional.
|     | ImmediateAntonyms |     |     | Pair |
| --- | ----------------- | --- | --- | ---- |
D:Thisisacasualmovie.
Q:MoviewhereTomHanksisrunningveryfast.
| Contrasting | MidAntonyms |     |     | Pair |
| ----------- | ----------- | --- | --- | ---- |
D:Inthismovie,TomHanksrunsmoderatelypaced.
Q:MoviewhereTomHanksisrunningveryfast.
|     | PolarAntonyms |     |     | Pair |
| --- | ------------- | --- | --- | ---- |
D:Inthismovie,TomHanksrunsveryslow.
Table3: Theproposedtaxonomyofnegationcategoriesandtheirformalization.
PromptforSententialNegation
Youareasystemthatreceivesadocument. Iwantyoutofollowthenextfoursteps:
1. Generateasearchquerythatcontainsexactlyonenegationword(’no’,’not’,or’none’).
Itshouldnotbeaccompaniedbyaquantifier.
Thequerymustbewell-definedandhaveafinite,verifiableanswerevenoutsidethedocument.
Avoidqueriesthatcouldhaveaninfinite,unboundedorexhaustivenumberofanswers.
Also,avoidqueriesthathavetheanswer’yes’or’no’.
Thequerymustbespecific,andsoundlikesomethingsomeonewouldtypeintoasearchengine.
2. Extractashortretrieval-stylepassagethatcontainsexactlyonenegationword(’no’,’not’,or
’none’).
- If the passage does not contain a negation, add exactly one negation word (’no’, ’not’, or
’none’).
3. Generatethepositiveversionofthesearchquerybyremovingthenegation.
4. Generatethepositiveversionofthepassagebyremovingthenegation. Keeptheotherwords
intact.
5. RespondinJSONformat.
Figure5: PromptsforSententialNegation
thegenerations,thesententialnegationshavebeen oneinstanceiscomposedofpairs<q ,doc >and
1 1
| filteredthemost. |     |     | <q ,doc >. |     |
| ---------------- | --- | --- | ---------- | --- |
2 2
| Statisticsofthegenerateddatasets. |     | Table4illus- |     |     |
| --------------------------------- | --- | ------------ | --- | --- |
A.4 Whatwedonotcover
tratesasummaryofthetwogenerateddatasets,i.e.,
thefreeandcontrolledgenerationdatasets. Length This section contains negation phenomena and
is calculated wrt. the number of words, while properties that, while interesting, we do not ac-
DataSizereferstothenumberofinstances,where

PromptforContrastingClausesYouareasystemthatreceivesadocument.Iwantyoutofollowthenextfoursteps.
Giventhefollowingdefinitionsoftypesofantonyms:
• Polarantonyms:Wordswithabsolute,directoppositemeaningwithnootherwordsbetweenthem.
• Midantonyms:Wordsdifferingslightly,notcompletelyopposed.
• Intermediateantonyms:Wordswithabsolute,directoppositemeanings,withmidantonymsbetweenthem.
Pickapairofmidantonymsthatmatchthisdocument.Namethemword1andword2.Avoidantonymsthathaveaprefix.
1. Generateasearchquerythatcontainsword1.Thequerymustbewell-definedandhaveafinite,verifiableanswerevenoutsidethedocument.Avoid
queriesthatcouldhaveaninfiniteorunboundednumberofanswers.Thequerymustbespecificandsoundlikesomethingsomeonewouldtypeinto
asearchengine.
2. Extractashortretrieval-stylepassagethatanswersthequeryandmustcontainword1.
3. Generatethepositiveversionofthesearchquerybyswitchingword1withword2.
4. Generatethepositiveversionofthepassagebyswitchingword1withword2.
RespondinJSONformat.
PromptforQuantifiers
Youareasystemthatreceivesadocument.Iwantyoutofollowthenextfoursteps.Generateonequery.Then,re-writeitinthefollowingstyles.Makesureall
querieshaveexactlythesamecontent:
1. Thefirstsearchquerymustuseexactlyoneuniversalquantifier(∀).
2. Thesecondsearchquerymustuseexactlyoneexistentialquantifier(∃),followedbyanegationinsideitsscope(∃x¬P(x)).Donotusetheword
’false’.
3. Thethirdsearchquerymustuseexactlyonenegation,followedbyanexistentialquantifier(∃)(¬∃xP(x)).Donotusetheword’false’
4. Thefourthsearchquerymustuseexactlyoneexistentialquantifier(∃),suchas“some”.Allqueriesmustbewell-definedandhaveafinite,verifiable
answer.Avoidqueriesthatcouldhaveaninfiniteorunboundednumberofanswers.Thequeriesmustbespecific,andsoundlikesomethingsomeone
wouldtypeintoasearchengine.Donotuseanysymbols.Extractashortretrieval-stylepassagethatanswersthefirstquery.Then,re-writeitinthe
followingstyles:
5. Thefirstpassagemustcontainexactlyoneuniversalquantifier(∀).
6. Thesecondpassagemustcontainexactlyoneexistentialquantifier(∃),followedbyanegationinsideitsscope(∃x¬P(x)).Donotusetheword
’false’.
7. Thethirdpassagemustcontainexactlyonenegation,followedbyanexistentialquantifier(∃)(¬∃xP(x)).
8. Thefourthpassagemustcontainexactlyoneexistentialquantifier(∃),suchas’some’.
9. "RespondinJSONformat."
|            |     |          | Figure6: | PromptsforContrastingClausesandQuantifiers |     |                |     |                            |     |     |     |
| ---------- | --- | -------- | -------- | ------------------------------------------ | --- | -------------- | --- | -------------------------- | --- | --- | --- |
|            |     |          |          |                                            |     | • Icallitluck, |     | but[itwould]not[havecomemy |     |     |     |
| Statistics |     | FreeGen. |          | Contr.Gen.                                 |     |                |     |                            |     |     |     |
wayhadInotbeenlookingoutforit].
| DataSize     |     | 1049/146/310 |     | 1031/143/305 |     |                                          |     |     |     |     |     |
| ------------ | --- | ------------ | --- | ------------ | --- | ---------------------------------------- | --- | --- | --- | --- | --- |
| Query1length |     | 10.25        |     | 10.20        |     |                                          |     |     |     |     |     |
|              |     |              |     |              |     | • Icallitluck,butitwouldnothavecomemyway |     |     |     |     |     |
| Query2length |     | 10.82        |     | 10.60        |     |                                          |     |     |     |     |     |
[hadI]not[beenlookingoutforit].
| Doc1length |                                      | 36.65 |     | 36.48 |      |                 |     |         |         |          |        |
| ---------- | ------------------------------------ | ----- | --- | ----- | ---- | --------------- | --- | ------- | ------- | -------- | ------ |
| Doc2length |                                      | 33.35 |     | 33.26 |      |                 |     |         |         |          |        |
|            |                                      |       |     |       |      | Scope analysis. |     | We also | exclude | analysis | on the |
| Table4:    | Statisticsofthetwogenerateddatasets. |       |     |       | Free |                 |     |         |         |          |        |
Gen.standsforfreegenerationdataset,whileControlled scope of the negation. In a sense, a query can be
|             |     |            |            |           |     | “Restaurants | that | do not | serve food” | and | the re- |
| ----------- | --- | ---------- | ---------- | --------- | --- | ------------ | ---- | ------ | ----------- | --- | ------- |
| Gen. stands | for | controlled | generation | dataset.T | The |              |      |        |             |     |         |
datasetsizeissplitintopartitions: train,validation,test. turneddocumentis“Restaurantsthatdonotwash
|     |     |     |     |     |     | laundry”. | To maintain |     | our study’s | focus, | we do |
| --- | --- | --- | --- | --- | --- | --------- | ----------- | --- | ----------- | ------ | ----- |
countforinthisstudy.
Inscopenon-negatedevents. notdelveintoscopeconsiderations. Moreover,the
Theseareexamples
scopeofnegationcanoftenshiftaccordingtocon-
ofeventsthatarenotnegated,despitebeingwithin
|           |      |          |         |               |     | text. For | example, | negation | can have | outer-read |     |
| --------- | ---- | -------- | ------- | ------------- | --- | --------- | -------- | -------- | -------- | ---------- | --- |
| the scope | of a | negation | Morante | and Daelemans |     |           |          |          |          |            |     |
andinner-reading,forexample“Itisnotlikelythat
| (2012). | Examples | are | shown | below. We | exclude |     |     |     |     |     |     |
| ------- | -------- | --- | ----- | --------- | ------- | --- | --- | --- | --- | --- | --- |
theYankeeswillwin.”:
thesecasesfromourstudy.
|     |     |     |     |     |     | • outer-reading: |     | (Likely...) | asin,itisnotprobable |     |     |
| --- | --- | --- | --- | --- | --- | ---------------- | --- | ----------- | -------------------- | --- | --- |
• Ishouldbegladtobeabletosayafterwardsthat
|     |     |     |     |     |     | thatitwillhappenthattheYankeeswillwin. |     |     |     |     | ¬∃  |
| --- | --- | --- | --- | --- | --- | -------------------------------------- | --- | --- | --- | --- | --- |
Ihadsolveditwithout[yourhelp].

| Variant |     | DifferencesinStep1andStep2 |     |     |     |     |     |     |     |     |     |     |
| ------- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Sentential Step1:Generateaquerythatcontainsexactlyonenegationword(’no’,’not’,or’none’).Itshouldnotbe
accompaniedbyaquantifier.Thequerymustbewell-definedandhaveafinite,verifiableanswereven
outsidethedocument. Avoidqueriesthatcouldhaveaninfinite,unboundedorexhaustivenumberof
answers.Also,avoidqueriesthathavetheanswer’yes’or’no’.Thequerymustbespecific,andsoundlike
somethingsomeonewouldtypeintoasearchengine.
Step2: Extractashortretrieval-stylepassagethatcontainsexactlyonenegationword(’no’,’not’,or
’none’).-Ifthepassagedoesnotcontainanegation,addexactlyonenegationword(’no’,’not’,or’none’).
Exceptor Step1:Generateasearchquerythatcontainsexactlyoneexclusionarywordsuchas(’others’,’besides’,
’but’,or’except’).Thequerymustbewell-definedandhaveafinite,verifiableanswerevenoutsidethe
document.Avoidqueriesthatcouldhaveaninfiniteorunboundednumberofanswers.Thequerymust
bespecific,andsoundlikesomethingsomeonewouldtypeintoasearchengine.
Step2:Extractashortretrieval-stylepassagethatanswersthequery. Makesurethepassagedoesnot
containanexclusionarywordsuchas(’others’,’besides’,’but’,or’except’).Makesurethepassagealso
containstheexcludedpartfromthequery.
Affixal Step1:Generateasearchquerythatcontainsexactlyoneaffixalnegationsuchas(’un-’,’in-’,’im-’,’il-’,
’ir-’,’dis-’,’non-’,’mis-’,’ill-’). Anaffixalnegationaddsaprefixorsuffixtoreversethemeaningofa
word.Thequeryshouldnotcontainanyothernegation.Thequerymustbewell-definedandhaveafinite,
verifiableanswerevenoutsidethedocument.Avoidqueriesthatcouldhaveaninfiniteorunbounded
numberofanswers. Thequerymustbespecific,andsoundlikesomethingsomeonewouldtypeintoa
searchengine.
Step2:Extractashortretrieval-stylepassagethatanswersthequery.-Inansweringthequery,thepassage
mustcontainexactlythesameaffixalnegationasinthequery.-Ifthepassagedoesnotcontainanaffixal
word,addexactlythesameoneasinthequery.Thepassageshouldnotcontainanyothernegation.
Implicit Step1:Generateasearchquerythatcontainsexactlyoneimplicitnegation.Animplicitnegationisone
thatdoesnotcontainanegationoperator.Theworditselfhasnegativesemantics.Examplesare(’avoid’,
’refuse’,’deny’,’ignore’).Itdoesnotincludeaffixalnegations.Thequeryshouldnotcontainanyother
negation.Thequerymustbewell-definedandhaveafinite,verifiableanswerevenoutsidethedocument.
Avoidqueriesthatcouldhaveaninfiniteorunboundednumberofanswers.Thequerymustbespecific,
andsoundlikesomethingsomeonewouldtypeintoasearchengine.
Step2:Extractashortretrieval-stylepassagethatanswersthequery.-Inansweringthequery,thepassage
mustcontainexactlythesameimplicitnegationasinthequery. -Ifthepassagedoesnotcontainthe
implicitnegation,addityourself.Thepassageshouldnotcontainanyothernegation.
Figure7: Summaryofdifferencesinpromptvariantsfordifferenttypesofnegation.
|     |     |     |     |     |     |     | A.5 LMLogicclassification |                 |                             |          |          |            |
| --- | --- | --- | --- | --- | --- | --- | ------------------------- | --------------- | --------------------------- | -------- | -------- | ---------- |
|     |     |     |     |     |     |     | When applying             |                 | the typed                   | lambda   | calculus | for-       |
|     |     |     |     |     |     |     | malization                | categorization, |                             | we check |          | both pairs |
|     |     |     |     |     |     |     | (q ,doc                   | )and(q          | ,doc )forthepresenceofnega- |          |          |            |
|     |     |     |     |     |     |     | 1                         | 2               | 2 1                         |          |          |            |
tion,asaresultofnotknowingnecessarilywhere
|     |     |     |     |     |     |     | negation | is present. | For | example, | NevIR | is con- |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | --- | -------- | ----- | ------- |
Figure8: Distributionofnegationtypes. structedsuchthatnegationisalwayspresentinthe
• inner-reading: Likely ... as in, it is likely the first pair, while ExcluIR is constructed such that
Yankeeswillnotwin. ∃¬x negationisalwayspresentinthesecondpair. Our
|             |                                   |            |     |             |     |        | classification | mechanism |     | is robust | to  | such varia- |
| ----------- | --------------------------------- | ---------- | --- | ----------- | --- | ------ | -------------- | --------- | --- | --------- | --- | ----------- |
| Litotes.    | Doublenegationdoesnotalwaysreduce |            |     |             |     |        | tions.         |           |     |           |     |             |
| to x, i.e., | not                               | not x does | not | necessarily |     | mean x |                |           |     |           |     |             |
(Horn, 2010). Such figure of speech is called a A.6 AnnotatorsTemplate
litotes,whereanunderstatementismadebyadding The queries and documents have been shuffled
| anegative.                           | Examplecanbe: |     |       |        |     |        |                           |                 |     |            |              |         |
| ------------------------------------ | ------------- | --- | ----- | ------ | --- | ------ | ------------------------- | --------------- | --- | ---------- | ------------ | ------- |
|                                      |               |     |       |        |     |        | within the                | instance,       | and | the sample | used         | for an- |
|                                      |               |     |       |        |     |        | notations                 | has a perfectly |     | balanced   | distribution | of      |
| • Idon’tdislikecars.                 |               |     | (¬∀¬x | = ∃¬¬x | =   | ∃x)can |                           |                 |     |            |              |         |
|                                      |               |     |       |        |     |        | labels. Givenaninstance(q |                 |     | ,doc       | )and(q       | ,doc ), |
|                                      |               |     |       |        |     |        |                           |                 |     | 1          | 1            | 2 2     |
| beseenasanunderstatementofIlikecars. |               |     |       |        |     | (∀x)   |                           |                 |     |            |              |         |
weaskthefollowingquestionstotheannotators:
| • Notbad!   | isanunderstatementofGood! |           |         |           |        |         |                                       |        |     |     |     |     |
| ----------- | ------------------------- | --------- | ------- | --------- | ------ | ------- | ------------------------------------- | ------ | --- | --- | --- | --- |
|             |                           |           |         |           |        |         | Q1: Whichdocumentismorerelevantforq1? |        |     |     |     |     |
| Existential | quantifiers               |           | with    | different |        | scopes. |                                       |        |     |     |     |     |
|             |                           |           |         |           |        |         |                                       | • doc1 |     |     |     |     |
| Quantifiers | such                      | as        | “every” | and       | “some” | apply   |                                       |        |     |     |     |     |
|             |                           |           |         |           |        |         |                                       | • doc2 |     |     |     |     |
| different   | scopes:                   | Every     | man     | didn’t    | win.   | Some    |                                       |        |     |     |     |     |
|             |                           |           |         |           |        |         |                                       | • none |     |     |     |     |
| man didn’t  | win.                      | ∀x(Man(x) |         | →         | ¬W(x)) | and     |                                       |        |     |     |     |     |
|             |                           |           |         |           |        |         |                                       | • both |     |     |     |     |
∃x(Man(x)∧¬W(x)).

SystemPrompt
1. YouareaMontagoviansemanticistworkinginatypedλ-calculusframework.
2. Foreachinputquery,followthenextfoursteps:
1. LEXICON:Listeverypredicateandquantifierasaλ-termwithanexplicitChurchtypeannotation.
2. SEMANTICINVENTORY:Outputtwocomma-separatedlists:
• Predicates:[]
• Quantifiers:[∃,∀]
3. NEGATIONANALYSIS:Foreachpredicate,indicatewhetheritmatchesoneofthefollowingcategories:
• Sentential(e.g.no,not,none,never,cannot)
• Exclusionary(e.g.besides,except,but)
• Affixal(e.g.boundmorphemesim-,in-,un-,-less,etc.)
• Implicit(e.g.verbssuchasdeny,refuse,avoid,fail)
4. FINALFORMULA:Presentthefullyreducedλ-termforS,oranequivalentfirst-orhigher-orderlogic
formula,enclosedinafencedcodeblock.
3. RespondinJSONformat.
4. Example:
Query:
Whatorganismsbesidescyanobacteriaperformanoxygenicphotosynthesis?
LEXICON:
organism:λx:e.Organism(x),
cyanobacteria:λx.Cyanobacteria(x),
perform_anoxygenic_photosynthesis:λx.PerformAnoxygenicPhotosynthesis(x),
besides:λPQx.Q(x)∧¬P(x)
SEMANTICINVENTORY:
Predicates:[Organism,Cyanobacteria,PerformAnoxygenicPhotosynthesis],Quantifiers:[∃]
NEGATIONANALYSIS:
Sentential:[],Exclusionary:[besides],Affixal:[],Implicit:[]
FINALFORMULA:
λx: e.Organism(x) ∧ PerformAnoxygenicPhotosynthesis(x) ∧ ¬Cyanobacteria(x)
Figure9: Promptforgeneratingtypedlambdacalculusproofs.
Q2: Whichdocumentismorerelevantforq2? 3: Minorissues
4: Languageflowswell
• doc1
5: Perfectlypolished
• doc2
• none Q5: Ratethecoherence(logicalflow)ofthetext.
• both
1: Nologicalflow[e]
Q3: Whichinstancescontainnegation? Multi- 2: Significantlogicalgaps
plechoicesarepossible. 3: Basiclogicalstructure
NOTE: If the individual instances do not 4: Generallylogicalandclear
containnegation,butthepair(q1,q2)con- 5: Completelylogicalandclear
tains antonyms, check both q1 and q2.
Q6: Ratetheconsistencyofinformationinthe
Samegoesfor(doc1,doc2).
text.
♢ q1
1: Contradictory
♢ q2
2: Unstable
♢ doc1
3: Mixed
♢ doc2
4: Aligned
Q4: Ratethenaturalness(fluencyandreadabil- 5: FullyAligned
ity)ofthetext.
A.6.1 Statisticalanalysisonannotationresults
1: Textisforced Table5showstheperformanceofannotatorswith
2: Noticeablyawkward respecttothegroundtruthlabelsofthegenerated

|     | T1  |     | T2  | T3  |     | T4  | T5 T6 | T7 T8 | T9 T10 |
| --- | --- | --- | --- | --- | --- | --- | ----- | ----- | ------ |
q1 0.79±0.21 0.64±0.21 0.79±0.07 0.71±0.14 0.86±0.00 0.79±0.07 0.79±0.07 0.79±0.07 0.79±0.07 0.64±0.21
q2 0.79±0.07 0.21±0.07 0.93±0.07 0.71±0.00 0.79±0.07 0.79±0.07 0.71±0.00 0.79±0.07 0.79±0.07 0.57±0.14
q3 0.91±0.04 1.00±0.00 0.90±0.04 0.96±0.03 0.94±0.01 0.87±0.03 0.90±0.08 0.81±0.00 0.77±0.14 0.69±0.07
q4 3.86±0.00 3.71±0.37 4.29±0.57 3.79±0.21 4.21±0.21 4.29±0.14 4.07±0.18 4.36±0.07 4.21±0.07 4.29±0.29
q5 3.86±0.14 4.21±0.24 4.07±0.36 3.57±0.14 4.14±0.00 4.29±0.14 4.14±0.14 4.29±0.00 4.21±0.21 4.07±0.21
q6 3.86±0.29 4.21±0.26 4.50±0.50 4.57±0.14 4.29±0.00 3.71±0.57 3.79±0.36 4.50±0.36 3.79±0.79 3.93±0.36
Table5: Performanceofannotatorswithrespecttothegroundtruthlabelsonthegeneratedquery-documentpairsof
bothsyntheticallygenerateddocuments. Eachscorerepresentsameanwithanstd. erroroverthetwodatasets.
|     | T1  |     | T2  | T3  |     | T4  | T5 T6 | T7 T8 | T9 T10 |
| --- | --- | --- | --- | --- | --- | --- | ----- | ----- | ------ |
q1 0.60±0.02 0.26±0.17 0.89±0.11 0.58±0.18 0.52±0.20 0.65±0.35 0.52±0.12 0.90±0.11 0.53±0.01 0.56±0.03
q2 0.58±0.02 0.30±0.02 0.86±0.14 0.53±0.01 0.89±0.11 0.57±0.21 0.31±0.20 0.90±0.11 0.55±0.02 0.58±0.22
q3 0.78±0.11 1.00±0.00 0.93±0.01 1.00±0.00 0.92±0.08 0.74±0.16 0.67±0.08 0.85±0.05 0.87±0.13 0.87±0.02
q4 0.80±0.01 0.30±0.20 0.71±0.29 0.52±0.08 0.79±0.21 0.79±0.21 0.49±0.14 0.76±0.24 0.76±0.04 0.89±0.11
q5 0.75±0.26 0.30±0.20 0.68±0.32 0.63±0.37 0.89±0.11 0.76±0.02 0.69±0.10 0.64±0.09 0.71±0.29 0.37±0.01
q6 0.55±0.02 0.36±0.30 0.67±0.05 0.36±0.36 0.33±0.40 0.44±0.28 0.31±0.13 0.78±0.22 0.56±0.20 0.56±0.22
Table 6: Inner Agreement of annotators on their answers about the generated query-document pairs of both
syntheticallygenerateddocuments. Eachscorerepresentsameanwithanstd. erroroverthetwodatasets.
datasets,i.e.,averagedoverboththefreeandcon- acrossthetwodatasets.
| trolled | generation |     | datasets. |           | The rows | q1-q6  | in-         |     |     |
| ------- | ---------- | --- | --------- | --------- | -------- | ------ | ----------- | --- | --- |
|         |            |     |           |           |          |        | A.7 Results |     |     |
| dicate  | the        | six | questions | presented |          | to the | annota-     |     |     |
tors, and the columns T1-T10 present the results In Figures 10, 11 and 12 we illustrate a close-up
oftheiranswerssplitacrossthetentypesofnega- of the free generation synthetic experiments, the
tionpresentinthesampleshowntotheannotators. controlledgenerationexperiments,andevaluation
onExcluIRasaresultofourcategorizationmecha-
| Forabriefdescriptionofthequestions:       |     |     |     |     |     | q1-q2ask |       |     |     |
| ----------------------------------------- | --- | --- | --- | --- | --- | -------- | ----- | --- | --- |
| abouttherelevanceofthetwodocumentsforeach |     |     |     |     |     |          | nism. |     |     |
query,andareassessedthroughaccuracy;q3asks
| about   | the       | presence | of         | negation | in the | generation |       |     |     |
| ------- | --------- | -------- | ---------- | -------- | ------ | ---------- | ----- | --- | --- |
| (binary | question; |          | therefore, | it       | does   | not ask    | about |     |     |
thespecifictypeofnegation)andisassessedusing
| the | f1 score; | q4-a6 | are | questions | about | the | logic, |     |     |
| --- | --------- | ----- | --- | --------- | ----- | --- | ------ | --- | --- |
naturalness,andconsistencyofinformationinthe
generatedqueriesanddocuments,andareassessed
bytakinganaverageoftheanswersrepresentedon
anordinalscalefrom1-5.
|          | Table | 6 shows | the inner | agreement |           | of   | the an- |     |     |
| -------- | ----- | ------- | --------- | --------- | --------- | ---- | ------- | --- | --- |
| notators |       | when    | answering | the       | questions | wrt. | the     |     |     |
twogenerateddatasets,i.e.,averagedoverboththe
| freeandcontrolledgenerationdatasets. |          |     |             |           |           | Therows |        |     |     |
| ------------------------------------ | -------- | --- | ----------- | --------- | --------- | ------- | ------ | --- | --- |
| q1-q6                                | indicate |     | the six     | questions | presented |         | to the |     |     |
| annotators,                          |          | and | the columns |           | T1-T10    | present | the    |     |     |
resultsoftheiranswerssplitacrossthetentypesof
negationpresentinthesampleshowntotheannota-
| tors. | Forabriefdescriptionofthequestions: |     |     |     |     |     | q1-q2 |     |     |
| ----- | ----------------------------------- | --- | --- | --- | --- | --- | ----- | --- | --- |
askabouttherelevanceofthetwodocumentsfor
eachquery,andtheagreementismeasuredusing
Cohen’sKappa;q3asksaboutthepresenceofnega-
tioninthegeneration(binaryquestion;therefore,
itdoesnotaskaboutthespecifictypeofnegation)
| and | is assessed |     | using recall |     | of agreement; |     | q4-a6 |     |     |
| --- | ----------- | --- | ------------ | --- | ------------- | --- | ----- | --- | --- |
arequestionsaboutthelogic,naturalness,andcon-
| sistency |     | of information |     | in the | generated |     | queries |     |     |
| -------- | --- | -------------- | --- | ------ | --------- | --- | ------- | --- | --- |
anddocuments,andareassessedusingaweighted
| Cohen’s |       | Kappa, | given | the answers |        | represent    | an  |     |     |
| ------- | ----- | ------ | ----- | ----------- | ------ | ------------ | --- | --- | --- |
| ordinal | scale | from   | 1-5.  | The         | scores | are averaged |     |     |     |

Figure10: Close-upofresultsontheFreeGeneration.

Figure 11: Pairwise Accuracy on the Controlled Generations dataset. Each column represents a negation type
followingourtaxonomy,includingtheFulldatasetinthefirstcolumn. Eachmodelisrepresentedbyonerow.

Figure12: PairwiseAccuracyonExcluIR.ThedatasetissplitwithoutclassificationMechanism.

A.7.1 Finetuningcurves
Figures13and14illustratethefine-tuningcurves
| for ColBERT,  | MultiQA       |            | and Mistral | when fine- |
| ------------- | ------------- | ---------- | ----------- | ---------- |
| tuned         | on synthetic, | NevIR,     | and a mix   | of the     |
| two datasets. | The           | evaluation | is done     | on NevIR   |
| with pairwise | accuracy,     | and        | on MSMarco  | with       |
MRR@10.
0.8
0.7
0.6
ycaruccA esiwriaP
0.5
0.4
0.3
0.2
0.1
|                  | 0.0 2.5                                     | 5.0 7.5 Epo1c0h.0                           | 12.5 15.0 17.5                              |     |
| ---------------- | ------------------------------------------- | ------------------------------------------- | ------------------------------------------- | --- |
|                  | ColBERT NevIR                               | MultiQA NevIR                               | Mistral NevIR                               |     |
|                  | ColBERT Synthetic ColBERT NevIR + Synthetic | MultiQA Synthetic MultiQA NevIR + Synthetic | Mistral Synthetic Mistral NevIR + Synthetic |     |
| Figure13:        | Fine-tuningresultsforColBERTandMul-         |                                             |                                             |     |
| tiQAon3datasets: |                                             | NevIRtrain,freegenerationtrain,             |                                             |     |
| andMixed.        | EvaluatedagainstNevIRdev.                   |                                             |                                             |     |
0.7
0.6
01@RRM 0.5
0.4
0.3
0.2
0.1
|                  | 0.00.0 2.5                                  | 5.0 7.5 Epo1c0h.0                           | 12.5 15.0 17.5                              |     |
| ---------------- | ------------------------------------------- | ------------------------------------------- | ------------------------------------------- | --- |
|                  | ColBERT NevIR                               | MultiQA NevIR                               | Mistral NevIR                               |     |
|                  | ColBERT Synthetic ColBERT NevIR + Synthetic | MultiQA Synthetic MultiQA NevIR + Synthetic | Mistral Synthetic Mistral NevIR + Synthetic |     |
| Figure14:        | Fine-tuningresultsforColBERTandMul-         |                                             |                                             |     |
| tiQAon3datasets: |                                             | NevIRtrain,freegenerationtrain,             |                                             |     |
| andMixed.        | EvaluatedagainstMSMarcodev.                 |                                             |                                             |     |
