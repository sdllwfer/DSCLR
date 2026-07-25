CoDeR: Local Constraint-Compatible Retrieval Beyond Semantic
Similarity
XingkunYin1 XuebinTang2 HongyangDu1*
1DepartmentofElectricalandComputerEngineering,UniversityofHongKong
HongKongSAR,China
2SchoolofComputerandCommunicationEngineering,UniversityofScienceandTechnologyBeijing
Beijing,China
{yinxingkun@connect.,duhy@eee.}hku.hk tangxb@xs.ustb.edu.cn
Abstract
User query
I need a quiethotel for light sleep, a still place away from
Informationretrievalsystemshavelongtreated noisy nightlife.
semantic similarity as a proxy for relevance.
For constraint-sensitive queries, this proxy
Similarity Retriever CoDeRRetriever
can fail when a document is topically close ranks by lexical and topical overlap distinguishes antonyms / negation
to the query but supports the opposite con-
straintdirection,suchassatisfyinganattribute Misleading evidence Compatible evidence
that should be excluded or affirming a rela- Hotel A Review Hotel B Review
tion that should be negated. We study this Hotel A, unlike other quiet Hotel B is a courtyard
hotels for light sleep, hotel that offers stillback
failure as constraint-violating evidence expo-
features noisynightlife rooms with soundproofing
sure and propose CoDeR, a local constraint- appreciated by tourists. for a quietstay.
compatible dense retrieval method that sepa-
× ✓
ratestopicalrelevancefromconstraintcompat-
ibility. CoDeR keeps a standard topical en- Topically relevant but Topically relevant and
constraint violating constraint compatible
coderforcandidatecoverageandaddsacom-
patibilityscorer,implementedasabi-encoder, Top retrieval output Top retrieval output
trainedwithlexical-polaritysupervisionover Hotel A Hotel B
contrastivesatisfyingandviolatingevidences.
Wrong evidence exposed Compatible evidence
Thecompatibilitysignalcanbeusedtorescore for downstream exposed for downstream
topical candidates or to retrieve an auxiliary
compatibility-orientedcandidateset,producing Figure1:Exampleofconstraint-violatingevidenceexposure.
Atopicallyrelevantreviewcanrepeatthequery’sexcluded
arankeddocumentlistwithoutexternalLarge
conditionandthereforesupportthewrongconstraintdirection,
LanguageModel(LLM)callsatinferencetime.
eventhoughitoverlapsstronglywiththequery.
WeevaluateCoDeRoncontrolleddiagnostics
andpublicnegative-constraintretrievalbench-
marks. Acrossthreecontrolleddiagnosticsets 2020). Existingdenseretrieversareoptimizedpri-
targeting antonymy, negation, and exclusion, marilyforsemanticsimilarity,wheretheyreward
CoDeR reduces V@2 by 20.59, 23.53, and
documents that are close to the query in mean-
5.77pointsrelativetothestrongestnon-CoDeR
ing (Karpukhin et al., 2020; Izacard et al., 2021;
baselines, andimprovesFVRbypushingthe
Xiao et al., 2024) rather than documents that sat-
firstviolatingdocumentdeeperintheranking.
isfytheintendedconstraintdirection. Thisdesign
Oursourcecodeanddatasetsareavailableat
https://github.com/NICE-HKU/CoDeR. becomesfragilewhenthequerycontainsexplicit
constraints, preferences, or exclusions. For such
1 Introduction constraint-sensitivequeries,relevanceisdirectional
ratherthanmerelytopical. Adocumentmaymen-
Informationretrieval(IR)systemsareincreasingly
tionthesameentities,attributes,anddomainvocab-
usedinreal-worldapplicationswhereusersmust
ularyasthequery,yetdescribetheconditionthat
locate evidence from large, domain-specific, or
theuserwantstoavoid. Thus,thehighest-ranked
corpus-specific document collections. In such
document can be semantically close while being
systems, risk begins at retrieval because the re-
constraint-incompatible,makingitaplausiblebut
triever decides which documents are exposed to
harmfulretrievalresult.
users,readers,ordownstreammodels(Lewisetal.,
Thisfailurecanariseacrossreal-worldretrieval
*Correspondingauthor settings where user intent is expressed through
1
6202
nuJ
11
]RI.sc[
1v40231.6062:viXra

constraints, including medical retrieval with ex- trievalquality. OnNevIRandExcluIR,CoDeRob-
clusion conditions, legal search, product recom- tainsthestrongestviolation-orientedperformance,
mendation, travel and hotel preferences, and en- suggestingthatlocalcompatibilityscoringcanre-
terpriseknowledge-basequestionanswering. Fig- duceearlyviolationexposurerelativetosemantic-
ure1illustratesonesuchcasewithahotel-search similarity-basedretrieval.
query that asks for a quiet stay while avoiding Ourmaincontributionsareasfollows:
| noisy nightlife. |     | A standard |     | retriever | may | rank a |      |           |                      |     |     |          |
| ---------------- | --- | ---------- | --- | --------- | --- | ------ | ---- | --------- | -------------------- | --- | --- | -------- |
|                  |     |            |     |           |     |        | • We | formulate | constraint-violating |     |     | evidence |
reviewofHotelAaboveamoresuitablereviewof
Hotel B because it repeats the query’s key terms, exposureasaretrieval-sidefailuremodefor
even though it describes the noise the user wants constraint-sensitive queries, and evaluate it
withV@kandFVRdiagnostics.
| to avoid.        | By  | contrast, | the  | Hotel B  | review | pro-  |      |         |        |     |         |             |
| ---------------- | --- | --------- | ---- | -------- | ------ | ----- | ---- | ------- | ------ | --- | ------- | ----------- |
| vides compatible |     | cues      | such | as quiet | back   | rooms |      |         |        |     |         |             |
|                  |     |           |      |          |        |       | • We | propose | CoDeR, |     | a local | constraint- |
andsoundproofing,butoverlapslessdirectlywith
|                         |                                    |     |            |     |                |     | compatible |           | retrieval | method             | that | composes |
| ----------------------- | ---------------------------------- | --- | ---------- | --- | -------------- | --- | ---------- | --------- | --------- | ------------------ | ---- | -------- |
| thequery.               | Werefertosuchtopicallyplausiblebut |     |            |     |                |     |            |           |           |                    |      |          |
|                         |                                    |     |            |     |                |     | topical    | retrieval |           | with compatibility |      | signals  |
| constraint-incompatible |                                    |     | candidates |     | as constraint- |     |            |           |           |                    |      |          |
learnedfromlexical-polaritycontrasts.
violatingevidence.
Thispaperstudiesconstraint-violatingevidence
• WeevaluateCoDeRoncontrolleddiagnostic
| exposure | as a | retrieval-side |     | failure mode. |     | We in- |      |            |                     |     |     |           |
| -------- | ---- | -------------- | --- | ------------- | --- | ------ | ---- | ---------- | ------------------- | --- | --- | --------- |
|          |      |                |     |               |     |        | sets | and public | negative-constraint |     |     | retrieval |
vestigatewhetheraretrievercanreduceearlyexpo-
benchmarks,showingreducedearlyviolation
suretoconstraint-violatingevidencewhilepreserv-
exposurewhilemaintainingcompetitivetopi-
| ing topical | coverage, |     | and | measure | this risk | with |     |     |     |     |     |     |
| ----------- | --------- | --- | --- | ------- | --------- | ---- | --- | --- | --- | --- | --- | --- |
calretrievalquality.
| violation-oriented |     | diagnostics |     | such | as V@k | and |     |     |     |     |     |     |
| ------------------ | --- | ----------- | --- | ---- | ------ | --- | --- | --- | --- | --- | --- | --- |
FVR.Recentworkaddressesnegative-constraintre-
|     |     |     |     |     |     |     | 2 RelatedWork |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- |
trievalthroughlogicalreasoningpipelinesorquery-
sideembeddingoptimization(Xuetal.,2025;Lee Dense retrievers encode queries and documents
et al., 2026). Whether local retrieval can directly intoasharedembeddingspaceandrankdocuments
score document-level constraint compatibility re- bylexicalorsemanticsimilarity(Karpukhinetal.,
mainsunderexplored. 2020;Izacardetal.,2021;Xiongetal.,2020). This
|            |     |        |     |                   |     |     | paradigm | is effective |     | for topical | matching, | but it |
| ---------- | --- | ------ | --- | ----------------- | --- | --- | -------- | ------------ | --- | ----------- | --------- | ------ |
| We propose |     | CoDeR, | a   | local information |     | re- |          |              |     |             |           |        |
trieval method that decouples topical relevance canfailwhentheuser’sintentdependsonpolarity,
fromconstraintcompatibility. Constraint-sensitive negation,orexclusion(Welleretal.,2024;Zhang
retrievalmustpreservetopicalcoveragewhilesup- etal.,2025). Recentworkhasthereforeexplored
pressingdocumentsthataretopicallyplausiblebut retrievalmethodsthatmodifythequery,document
constraint-incompatible. CoDeRretainsastandard representation,orscoringproceduretobettercap-
topicalencoderforquery-intentmatchingandintro- ture complex intent (Gao et al., 2023). CoDeR
ducesacompatibilityencoderthatscoreswhether followsthisembedding-sideretrievalperspective,
a candidate document satisfies the constraint ex- but focuses on separating topical relevance from
pressed in the query. Trained with lightweight constraintcompatibility.
lexical-polaritysupervisionoverantonymy,nega- The closest line of work studies negative-
tion,andexclusion,thisencoderseparatessatisfy- constraint and negation-aware retrieval. NS-IR
ing documents from nearby documents with the introducesNegConstraintandrerankscandidates
opposite constraint direction. A modular integra- throughlogicalconsistencyaftertranslatingqueries
tionpolicycombinesthetopicalandcompatibility and documents into first order logic (Xu et al.,
signals to produce a compatibility-aware list. At 2025). DEOdecomposesnegationawarequeries
inference time, CoDeR uses only local encoder intopositiveandnegativecomponentsanddirectly
scoringwithoutexternalLLMcalls. optimizesqueryembeddingswithouttraining(Lee
We evaluate CoDeR on controlled diagnos- et al., 2026). Both methods address important
tics and public negative-constraint benchmarks. limitationsofstandardneuralretrievers,butdiffer
Across antonymy, negation, and exclusion diag- fromCoDeRindeploymentpathandscoringtarget.
nostics, CoDeR lowers V@2 from 73.53, 72.55, Logicbasedpipelinesoftenrequiretranslationor
and 53.85 to 52.94, 49.02, and 48.08, respec- judgingmodulesthatincreaselatencyanddeploy-
tively, while maintaining competitive topical re- ment complexity when implemented with strong
2

externalLLMs,whilequeryembeddingoptimiza- Weexpressthisdesideratumas
| tion is lightweight |     | but | mainly | changes |     | the query |         |     |          |     |     |     |     |
| ------------------- | --- | --- | ------ | ------- | --- | --------- | ------- | --- | -------- | --- | --- | --- | --- |
|                     |     |     |        |         |     |           | f(q,d+) |     | f(q,d−), | d+  |     | d−  |     |
representation. CoDeRinsteadlearnsalocalcon- > ∈ S , ∈ V , (2)
|     |     |     |     |     |     |     |     |     |     |     |     | q   | q   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
straintcompatibilityscorerthatdirectlyevaluates
candidatedocumentsandproducesacompatibility- wheref denotesthefinalretrievalscore.
aware ranked list without external LLM calls at Weevaluate whetherviolating evidenceenters
theretrievedcontextusingretrieval-sidediagnos-
inferencetime.
|                                            |     |     |     |     |     |     | tics.                | Let R¯ | (q) = | {r : 1                  | ≤ j | ≤ k} be | the set |
| ------------------------------------------ | --- | --- | --- | --- | --- | --- | -------------------- | ------ | ----- | ----------------------- | --- | ------- | ------- |
| Underlexicalpolarity,bothsparseanddensere- |     |     |     |     |     |     |                      |        | k     | j                       |     |         |         |
|                                            |     |     |     |     |     |     | ofreturneddocuments. |        |       | Theviolationindicatorat |     |         |         |
trieverscanbecomeunreliablebecausetheyreward
| lexicaloverlaporsemanticsimilarity.            |            |           |            |          | Antonymy,  |           | depthk  | is           |           |                |           |         |     |
| ---------------------------------------------- | ---------- | --------- | ---------- | -------- | ---------- | --------- | ------- | ------------ | --------- | -------------- | --------- | ------- | --- |
| negation,                                      | and        | exclusion | can        | make     | satisfying | and       |         |              |           |                |           |         |     |
|                                                |            |           |            |          |            |           |         |              |           | I(cid:2) R¯    |           | (cid:3) |     |
|                                                |            |           |            |          |            |           |         | V@k(q)       | =         | (q)∩V          |           | ̸= ∅ .  | (3) |
| violatingdocumentstopicallysimilar,turningcon- |            |           |            |          |            |           |         |              |           | k              |           | q       |     |
| straint                                        | violations | into      | structured |          | hard       | negatives |         |              |           |                |           |         |     |
|                                                |            |           |            |          |            |           | We      | also measure | the       | first          | violating | rank.   | Let |
| rather than                                    | ordinary   |           | irrelevant | passages |            | (Wasser-  |         |              |           |                |           |         |     |
|                                                |            |           |            |          |            |           | J k (q) | = {j         | ≤ k | r j | ∈ V q }∪{k+1}. |           |         |     |
| manetal.,2025;ChoandLee,2026).                 |            |           |            |          | Priorwork  |           |         |              |           |                |           |         |     |
useshardnegativestoexposeretrievalfailuresor
|                                              |     |                              |     |     |     |     |       |         | FVR (q) | = minJ         |     | (q).             | (4) |
| -------------------------------------------- | --- | ---------------------------- | --- | --- | --- | --- | ----- | ------- | ------- | -------------- | --- | ---------------- | --- |
| improvererankertraining(Quetal.,2021;Wasser- |     |                              |     |     |     |     |       |         | k       |                | k   |                  |     |
| manetal.,2025).                              |     | CoDeRinsteadtargetshardnega- |     |     |     |     |       |         |         |                |     |                  |     |
|                                              |     |                              |     |     |     |     | These | metrics | are     | retrieval-side |     | risk indicators. |     |
tivesfromconstraintviolationanduseslightweight
Theydonotbythemselvesdeterminedownstream
lexicalpolaritysupervisiontoseparatesatisfying
|     |     |     |     |     |     |     | answer | behavior, | but | they | measure | whether | the |
| --- | --- | --- | --- | --- | --- | --- | ------ | --------- | --- | ---- | ------- | ------- | --- |
evidencefromopposite-directionevidence.
top-rankedretrievalresultscontaindocumentsthat
conflictwiththeuser’sstatedconstraint.
3 ProblemFormulation
4 CoDeR:Constraint-Compatible
| Given a | query | q and | a document |     | corpus | D = |     |     |     |     |     |     |     |
| ------- | ----- | ----- | ---------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
Retrieval
}N
| {d  | , a retriever |     | assigns | each | d ∈ | D a score |     |     |     |     |     |     |     |
| --- | ------------- | --- | ------- | ---- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
i i=1
andreturnsthetopk documentsasarankedlist CoDeR instantiates the constraint-compatible re-
|     |     |     |     |     |     |     | trieval | objective | in  | Eq. 2 as | a local | information |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------- | --- | -------- | ------- | ----------- | --- |
R (q) = [r ,...,r ]. (1) retrievalmethod. AtopicalencoderE producesa
|     |     | k   | 1   | k   |     |     |                        |     |     |                        |     | T   |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | ---------------------- | --- | --- | --- |
|     |     |     |     |     |     |     | topicalrelevancescores |     |     | (q,d),whileaconstraint |     |     |     |
T
Standard retrievers typically rank documents by compatibilityencoderE producesacompatibil-
C
| lexical | overlap | or semantic |     | similarity. |     | This is in- |           |     |          |                |     |        |      |
| ------- | ------- | ----------- | --- | ----------- | --- | ----------- | --------- | --- | -------- | -------------- | --- | ------ | ---- |
|         |         |             |     |             |     |             | ity score | s   | C (q,d). | An integration |     | policy | com- |
sufficient for constraint-sensitive queries, where binesthesescorestoproduceacompatibility-aware
a document can be topically related to the query rankeddocumentlist. CoDeRoperatesbeforeany
whileviolatinganexplicituserrequirement.
optionaldownstreamrerankerorgeneratorandis
For analysis, we view a constraint-sensitive evaluatedbytherankedlistitdirectlyreturns.
| query as | containing |     | a topical | component |     | t and |     |     |     |     |     |     |     |
| -------- | ---------- | --- | --------- | --------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
q
4.1 ConstraintCompatibilityEncoder
| aconstraintcomponentc |     |     | q ,writtenasq |     |     | = (t q ,c q ). |     |     |     |     |     |     |     |
| --------------------- | --- | --- | ------------- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
Thisdecompositionisonlyusedtodefinethetask
|     |     |     |     |     |     |     | The | constraint | compatibility |     | encoder | E   | esti- |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------------- | --- | ------- | --- | ----- |
C
| and does | not assume |     | an explicit | symbolic |     | parser |     |     |     |     |     |     |     |
| -------- | ---------- | --- | ----------- | -------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
mateswhetheradocumentsatisfiestheconstraint
at inference time. For each query q, documents expressed in the query. Unlike a topical re-
areassignedquery-dependentevidencelabels. A triever, which mainly rewards semantic close-
| satisfying | document |     | d ∈ S | matches | the | topic t |      |        |        |        |         |                |     |
| ---------- | -------- | --- | ----- | ------- | --- | ------- | ---- | ------ | ------ | ------ | ------- | -------------- | --- |
|            |          |     |       | q       |     | q       | ness | to the | query, | E C is | trained | to distinguish |     |
and satisfies the constraint c . A violating docu- constraint-satisfyingevidencefromtopicallysim-
q
ment d ∈ V q matches the topic but violates the ilar constraint-violating evidence. In our imple-
| constraint. | Neutraltopicaldocumentsarerelated |     |     |     |     |     |            |     |        |             |     |           |     |
| ----------- | --------------------------------- | --- | --- | --- | --- | --- | ---------- | --- | ------ | ----------- | --- | --------- | --- |
|             |                                   |     |     |     |     |     | mentation, |     | E is a | bi-encoder, | so  | the query | and |
C
| tot butdonotexplicitlysatisfyorviolatec |     |     |     |     |     | . The |                                            |     |     |     |     |     |     |
| --------------------------------------- | --- | --- | --- | --- | --- | ----- | ------------------------------------------ | --- | --- | --- | --- | --- | --- |
| q                                       |     |     |     |     |     | q     | documentareencodedindependently(Reimersand |     |     |     |     |     |     |
samedocumentmaythereforebesatisfyingforone Gurevych,2019;Karpukhinetal.,2020). Givena
queryandviolatingforanother. queryq andadocumentd,itcomputesacompati-
| A constraint-compatible |     |     |     | retriever | should | pre- |     |     |     |     |     |     |     |
| ----------------------- | --- | --- | --- | --------- | ------ | ---- | --- | --- | --- | --- | --- | --- | --- |
bilityscorebyvectorsimilarity,
servetopicalcoveragewhilerankingsatisfyingev-
idenceabovetopicallysimilarviolatingevidence. s (q,d) = sim(E (q),E (d)). (5)
|     |     |     |     |     |     |     |     | C   |     | C   |     | C   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
3

0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.5 0.6 0.7 0.8 0.9
Topical score
erocs
ytilibitapmoC
tion,andexclusionpatternsthatchangethedirec-
tionofauserconstraintwhilepreservingthesur-
rounding topic. For each query q, we constructa
satisfyingdocumentd+ andaviolatingdocument
d−. Bothdocumentsremaintopicallyrelatedtothe
querywhichdiscouragesthemodelfromsolving
the task by topical matching alone. This design
concentratesthelearningsignalonconstraintcom-
patibilityratherthangeneraltopicalrelevance.
WeoptimizeE withamultiplenegativesrank-
C
Violating evidence (n=998) ingobjectiveovertriples(q,d+,d−)(Oordetal.,
Satisfying evidence (n=997)
2018). The satisfying document is treated as the
positivedocument, whiletheviolatingdocument
serves as an explicit hard negative. Other docu-
Figure2: TopicalityversuscompatibilityonExcluIR.Each mentsinthesamebatcharealsousedasin-batch
pointisalabeledcandidatefromthetopicalencoder’stop-k
negatives. Let N(q) denote the negative set for
results. Satisfyingandviolatingevidenceoverlapalongthe
topical-scoreaxisbutseparatealongthecompatibility-score queryq,includingtheexplicitviolatingdocument
axis,showingthatconstraintviolationsareoftentopicalnear
andin-batchnegatives. Thetraininglossis
neighborsratherthanoff-topicnoise.
L = −βs (q,d+)+logZ(q),
C
wheresim(·,·)denotescosinesimilaritybetween Z(q) = exp (cid:0) βs (q,d+) (cid:1)
C
the two embeddings. A higher value of s (q,d) (6)
C + (cid:88) exp (cid:0) βs (q,d−) (cid:1) .
indicates stronger evidence that d satisfies the re- C
quirementexpressedinq. d−∈N(q)
BecauseE isabi-encoder,documentembed-
C
whereβ isascalingfactor. Thisobjectiveencour-
dingscanbeprecomputedandindexedfortwouses
ages E to assign higher compatibility scores to
inCoDeR.Inthesequentialpolicy,E supplies C
C
satisfying evidence than to topically similar evi-
a compatibility score over the topical candidate
dencewiththewrongconstraintdirection.
set C (q). In the union policy, E retrieves a
T C
compatibility-drivencandidatesetC (q)fromthe
C
4.3 ModularIntegrationofCoDeR
samecorpus, allowingconstraint-compatibleevi-
dencethatmaybemissedbythetopicalretrieverto Anintegrationpolicyconvertsthetopicalandcom-
enterthecandidatepool. Thus,E complements patibility scores into the final ranked document
C
topicalretrievalwithaconstraint-compatibilitysig- list. Lets (q,d)denotethetopicalrelevancescore
T
nal,helpingseparatesatisfyingevidencefromtop- from E , and let s (q,d) denote the constraint
T C
ically similar violations. Implementation details, compatibility score from E . The policy may
C
includingthebaseencoder,queryprefix,andtrain- varyinhowcandidatesaregenerated,howthetwo
ingdatasplit,areprovidedinSection5.1andAp- scores are combined, and how low-compatibility
pendix B. Figure 2 illustrates why this separate candidatesarefiltered. Thismodulardesignisnot
compatibilitysignalisneeded. OnExcluIR,satis- tiedtoasingleimplementationandcansupportdif-
fying and violating evidence occupy overlapping ferentcandidategeneration,scoring,andfiltering
high-topicalityregionsunderthetopicalencoder, strategies. Inthispaper,weinstantiatethisdesign
indicatingthatviolationsarenotmerelyoff-topic withtwovariants,CoDeR-SeqandCoDeR-Union.
outliers. The trained compatibility encoder sepa- CoDeR-Seqisasequentialcompatibility-scoring
ratesthemalongthecompatibilityaxis,motivating policy that keeps the standard topical encoder as
CoDeR’s use of topical scoring for coverage and thecandidategeneratorandusesE torescoreand
C
compatibilityscoringforconstraintdirection. filtertopicalcandidates. CoDeR-Unionisaunion
candidatepolicythatletsbothE andE retrieve
T C
4.2 LexicalPolaritySupervision
candidates,thenmergesthetwocandidatesetsbe-
WetraintheconstraintcompatibilityencoderE forefusedranking. Thus,thetwovariantssharethe
C
withlightweightlexicalpolaritysupervision. The sametopicalandcompatibilityencoders,butdiffer
supervision is constructed from antonymy, nega- in whether thecompatibility encoder only scores
4

B. Union Candidate Retrieval AfterformingC U (q),CoDeRrescoreseverydoc-
|     | Query𝑞 | Corpus𝐷 |     |     |     |     |     |     |     |     |     |     |     |
| --- | ------ | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Topical Encoder Constraint Encoder ument in the union pool with both encoders. Be-
| A. Sequential Compatibility Scoring |      |                                 |     | retrieve 𝐶(cid:3021) |     | retrieve 𝐶(cid:3004) |         |     |       |      |           |       |         |
| ----------------------------------- | ---- | ------------------------------- | --- | -------------------- | --- | -------------------- | ------- | --- | ----- | ---- | --------- | ----- | ------- |
|                                     |      |                                 |     |                      |     |                      | cause s | and | s may | have | different | score | scales, |
|                                     |      |                                 |     |                      |     |                      | T       |     | C     |      |           |       |         |
|                                     | Topi | c a l   E n c o der 𝐸(cid:3021) |     | 𝐶(cid:3021)          |     | 𝐶(cid:3004) S* S* S* |         |     |       |      |           |       |         |
s c o r e  𝑠 𝑞 ,𝑑 S V N weuserank-basednormalizedscores. Letρ (q,d)
|     |     | (cid:3021) |     |     |     |     |     |     |     |     |     |     | T   |
| --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andρ (q,d)denotetherankpositionsofdinside
C
| Topical Candidate 𝐶(cid:3021)𝑞 |                                        | S V                               | N … | Union Pool𝐶(cid:3022)= 𝐶(cid:3021) ∪ 𝐶(cid:3004) |      |                         |               |            |     |     |        |      |         |
| ------------------------------ | -------------------------------------- | --------------------------------- | --- | ------------------------------------------------ | ---- | ----------------------- | ------------- | ---------- | --- | --- | ------ | ---- | ------- |
|                                |                                        |                                   |     |                                                  |      |                         | C (q) when    | candidates |     | are | sorted | by s | and s , |
|                                | topically similar, mixed compatibility |                                   |     | S                                                | S* V | N                       | U             |            |     |     |        | T    | C       |
|                                |                                        |                                   |     |                                                  |      |                         | respectively. | Wedefine   |     |     |        |      |         |
|                                | Const                                  | r a in t   E n c oder 𝐸(cid:3004) |     |                                                  |      | (cid:2869) (cid:2869)   |               |            |     |     |        |      |         |
|                                |                                        | s c o re   𝑠 (cid:3004) 𝑞 ,𝑑      |     | Rank Normalization                               |      | (cid:3096) , (cid:3096) |               |            |     |     |        |      |         |
(cid:3269) (cid:3252)
1
Co m p a tib il ity   F i lter Query -r e l a t iv e   Filter s (q,d) = ,
|     |     | V   |     |     |     | V   |     |     | (cid:101)T |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- |
k ee p  𝑠(cid:3004) 𝑞, 𝑑   ≥   𝜏 ke e p  𝑠(cid:3558) (cid:3004)   ≥  𝛾 (cid:3044) ρ (q,d)
|     |     |               |     |                          |     |     |     |     |     |     | T   |     | (10) |
| --- | --- | ------------- | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | ---- |
|     |     | Fused Scoring |     | Normalized Fused Scoring |     |     |     |     |     |     | 1   |     |      |
𝑓(cid:3046)(cid:3032)(cid:3044) = 𝛼𝑠(cid:3021)+(1 − 𝛼)𝑠(cid:3004) 𝑓(cid:3048)(cid:3041)(cid:3036)(cid:3042)(cid:3041) = 𝛼𝑠(cid:3558)(cid:3021)+(1 − 𝛼)𝑠(cid:3558)(cid:3004) s (q,d) = .
(cid:101)C
|     |     |     |     |     |     |     |     |     |     |     | ρ C (q,d) |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- |
Top-k compatibility-aware candidates
c o n s t ra i n t  v io la t i o n s   p u s h e d   d o w n S S* N V Thenormalizedscoresarethencombined:
|     | lo c a l  b i- e n c o | d e r   in f e r e n c e , n o |   e x ternal API calls |     |     | …   |     |     |     |     |     |     |     |
| --- | ---------------------- | ------------------------------ | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Figure3: OverviewofthetwoCoDeRintegrationpolicies. f union (q,d) = αs (cid:101)T (q,d)+(1−α)s (cid:101)C (q,d). (11)
| (A)SequentialIntegrationscorestopicalcandidateswithE |     |     |     |     |     |     | .   |     |     |     |     |     |     |
| ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
C
(B)UnionCandidateIntegrationmergescandidatesretrieved Theunionpolicyappliesaquery-relativecom-
| byE | andE | beforecompatibility-awareranking. |     |     |     |     |                |            |       |                  |            |     |           |
| --- | ---- | --------------------------------- | --- | --- | --- | --- | -------------- | ---------- | ----- | ---------------- | ---------- | --- | --------- |
|     | T    | C                                 |     |     |     |     | patibility     | filter.    | Given | a query-specific |            |     | threshold |
|     |      |                                   |     |     |     |     | γ , it retains | candidates |       | with             | s (q,d)    | ≥   | γ and     |
|     |      |                                   |     |     |     |     | q              |            |       |                  | (cid:101)C |     | q         |
topicalcandidatesoralsoretrievesadditionalcan- returnsthetopk documentsrankedbyEq.11. If
|     |     |     |     |     |     |     | no candidate | passes |     | the filter, | CoDeR | skips | only |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------ | --- | ----------- | ----- | ----- | ---- |
didatesfromthecorpus.
thehardcutoffandranksthefullunionpoolbythe
| SequentialIntegration |     |     |     | SequentialIntegrationis |     |     |            |        |            |     |               |     |       |
| --------------------- | --- | --- | --- | ----------------------- | --- | --- | ---------- | ------ | ---------- | --- | ------------- | --- | ----- |
|                       |     |     |     |                         |     |     | same fused | score, | preserving |     | compatibility |     | as an |
alightweightcompatibility-scoringpolicyshown activerankingsignal.
inFig.3(A).Itfirstretrievesatopicalcandidateset CoDeRreturnsarankeddocumentlistanddoes
| C   | (q) | E   |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
T using T , and then scores each candidate not modify any downstream reader, reranker, or
| withE | .   | Candidateswhosecompatibilityscores |     |     |     |     |            |      |            |     |        |     |           |
| ----- | --- | ---------------------------------- | --- | --- | --- | --- | ---------- | ---- | ---------- | --- | ------ | --- | --------- |
|       | C   |                                    |     |     |     |     | generator. | This | separation |     | allows | the | retrieval |
fallbelowathresholdτ arefiltered: method to be evaluated directly. Optional down-
streamcomponentscanbeattachedafterCoDeR,
|     | C   | (q) = {d | ∈ C (q) | : s (q,d) | ≥   | τ}. (7) |                                             |     |        |     |            |     |            |
| --- | --- | -------- | ------- | --------- | --- | ------- | ------------------------------------------- | --- | ------ | --- | ---------- | --- | ---------- |
|     | seq |          | T       | C         |     |         | whilethemainexperimentsisolatetheembedding- |     |        |     |            |     |            |
|     |     |          |         |           |     |         | side compatibility                          |     | signal | by  | evaluating |     | the direct |
Weranktheremainingcandidatesby:
rankedoutput.
|     | f seq (q,d) | = αs T | (q,d)+(1−α)s |     | C (q,d). | (8) | 5 Experiments |     |     |     |     |     |     |
| --- | ----------- | ------ | ------------ | --- | -------- | --- | ------------- | --- | --- | --- | --- | --- | --- |
WeevaluateCoDeRfromthreeretrieval-centered
| where | τ        | is the compatibility |                | threshold, |         | and α ∈ |                  |     |              |     |             |           |       |
| ----- | -------- | -------------------- | -------------- | ---------- | ------- | ------- | ---------------- | --- | ------------ | --- | ----------- | --------- | ----- |
|       |          |                      |                |            |         |         | perspectives:    |     | preservation |     | of ordinary | topical   | re-   |
| [0,1] | controls | the                  | balance        | between    | topical | rele-   |                  |     |              |     |             |           |       |
|       |          |                      |                |            |         |         | trieval quality, |     | reduction    | of  | early       | violation | expo- |
| vance | and      | constraint           | compatibility. |            | This    | policy  |                  |     |              |     |             |           |       |
sure,andevaluationonpublicnegative-constraint
| requires   |     | one topical | retrieval | call      | and    | local com- |             |     |      |        |                |     |       |
| ---------- | --- | ----------- | --------- | --------- | ------ | ---------- | ----------- | --- | ---- | ------ | -------------- | --- | ----- |
|            |     |             |           |           |        |            | benchmarks. | We  | also | report | inference-time |     | over- |
| patibility |     | scoring     | over C    | T (q). It | allows | an exist-  |             |     |      |        |                |     |       |
ing retrieval pipeline to add compatibility-aware head and API usage. A small downstream probe
isreportedinAppendixHasapreliminaryend-to-
rerankingwithoutchangingtheoriginalcandidate
endvalidation.
generationstage.
5.1 ExperimentalSetup
| UnionCandidateIntegration |     |     |     |     | UnionCandidate |     |     |     |     |     |     |     |     |
| ------------------------- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Integration,showninFig.3(B),relaxesthecover- Datasets and metrics. We use BEIR-style
agebottleneckofthesequentialpolicybyallowing datasets(Thakuretal.,2021)fortopical-retrieval
bothencoderstoparticipateincandidateretrieval. preservation,controlledantonymy/negation/exclu-
ThetopicalencoderretrievesC (q),whilethecon- sion diagnostics for constraint violations, public
T
straintcompatibilityencoderretrievescandidateset negative-constraint benchmarks for transfer. For
|     |     |     |     |     |     |     | ordinary | retrieval | preservation, |     | we  | report | nDCG |
| --- | --- | --- | --- | --- | --- | --- | -------- | --------- | ------------- | --- | --- | ------ | ---- |
C C (q). Thefinalcandidatepoolistheirunion:
andMAP.Forconstraint-awareretrieval,wereport
C (q) = C (q)∪C (q). (9) V@k and FVR: V@k measures whether at least
|     |     | U   | T   | C   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
5

oneconstraint-violatingdocumentappearsinthe competitivewithoutmodelingconstraintdirection.
topk results,whileFVRmeasurestherankofthe TherelevantobservationisthataddingE keeps
C
first violation. Lower V@k and higher FVR are CoDeRclosetothebasedenseretrieverwhilein-
better. FVRcapturestheearliestrankatwhichin- troducingacompatibilitysignalthatbecomesim-
compatibleevidenceappears,insteadofdepending portantwhentopicalrelevanceandconstraintsatis-
| onasinglefixedtop-k |                               |     | threshold(Liuetal.,2024).   |     |     |     | factiondiverge.                           |     |     |     |     |     |
| ------------------- | ----------------------------- | --- | --------------------------- | --- | --- | --- | ----------------------------------------- | --- | --- | --- | --- | --- |
| Baselines.          | Wegroupthecomparisonmethodsby |     |                             |     |     |     |                                           |     |     |     |     |     |
|                     |                               |     |                             |     |     |     | 5.3 Constraint-AwareRetrievalonDiagnostic |     |     |     |     |     |
| retrievalmechanism. |                               |     | Thelexicalretrievalbaseline |     |     |     | Datasets                                  |     |     |     |     |     |
isBM25,whichrepresentssparseterm-matching
Asgeneralretrievalbenchmarksdonottestwhether
| retrieval(RobertsonandZaragoza,2009). |     |     |     |     |     | Dense |     |     |     |     |     |     |
| ------------------------------------- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
retrieversseparatetopicalrelevancefromconstraint
semanticretrievalbaselinesincludeBGEandCon-
compatibility,weevaluateoncontrolledantonymy,
triever,whichrankdocumentsmainlybylearned
negation,andexclusiondatasetswheresatisfying
| semanticsimilarity.    |       | Thequery-generationbaseline |                          |                     |     |     |               |           |            |             |      |         |
| ---------------------- | ----- | --------------------------- | ------------------------ | ------------------- | --- | --- | ------------- | --------- | ---------- | ----------- | ---- | ------- |
|                        |       |                             |                          |                     |     |     | and violating | documents | are        | topical     | near | neigh-  |
| is HyDE,               | which | retrieves                   |                          | using LLM-generated |     |     |               |           |            |             |      |         |
|                        |       |                             |                          |                     |     |     | bors with     | opposite  | constraint | directions. |      | Table 2 |
| hypotheticaldocuments. |       |                             | Constraint-orientedbase- |                     |     |     |               |           |            |             |      |         |
reportsviolation-orientedmetricsforthison-topic
| lines include |     | NS-IR | and DEO, | the | methods | clos- |     |     |     |     |     |     |
| ------------- | --- | ----- | -------- | --- | ------- | ----- | --- | --- | --- | --- | --- | --- |
butconstraint-unsafesetting,withconstructionde-
| esttooursetting: |     | NS-IRtargetsconstraint-aware |     |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tailsandexamplesinAppendixA.
| retrieval,   | while               | DEO                 | is designed |            | for exclusion- |        |                 |           |            |            |               |      |
| ------------ | ------------------- | ------------------- | ----------- | ---------- | -------------- | ------ | --------------- | --------- | ---------- | ---------- | ------------- | ---- |
|              |                     |                     |             |            |                |        | The diagnostic  | results   | expose     |            | a progression | in   |
| oriented     | negative-constraint |                     |             | retrieval. | We             | report |                 |           |            |            |               |      |
|              |                     |                     |             |            |                |        | how different   | retrieval | mechanisms |            | handle        | con- |
| two proposed |                     | compatibility-aware |             |            | retrieval      | vari-  |                 |           |            |            |               |      |
|              |                     |                     |             |            |                |        | straints: BM25, | BGE,      | and        | Contriever | represent     |      |
ants,CoDeR-SeqandCoDeR-Union,correspond-
thebasicretrievalregime,wherelexicaloverlapor
| ing to the | sequential |     | and | union policies |     | in Sec- |     |     |     |     |     |     |
| ---------- | ---------- | --- | --- | -------------- | --- | ------- | --- | --- | --- | --- | --- | --- |
densesemanticsimilaritycanfinddocumentsabout
| tion4.3. | Allmethodsareevaluatedbytheirdirect |     |     |     |     |     |     |     |     |     |     |     |
| -------- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thesameentitiesandattributesbuthasnoexplicit
retrievaloutputwithoutanadditionalcross-encoder
reasontotreatsatisfyingandviolatingcounterparts
| reranker.           | Thissettingfollowsembedding-sidere- |     |              |         |     |         |                              |     |     |                 |     |     |
| ------------------- | ----------------------------------- | --- | ------------ | ------- | --- | ------- | ---------------------------- | --- | --- | --------------- | --- | --- |
|                     |                                     |     |              |         |     |         | asoppositeretrievaloutcomes. |     |     | Thismakesthedi- |     |     |
| trieval comparisons |                                     |     | and isolates | whether |     | the re- |                              |     |     |                 |     |     |
agnosticsettingdifficultastheviolatingdocument
| trieval method |     | itself | produces | a more | constraint- |     |                      |     |                         |     |     |     |
| -------------- | --- | ------ | -------- | ------ | ----------- | --- | -------------------- | --- | ----------------------- | --- | --- | --- |
|                |     |        |          |        |             |     | isnotoff-topicnoise, |     | butatopicalnearneighbor |     |     |     |
compatiblerankedlist.
pointinginthewrongconstraintdirection.
Implementation details. The constraint HyDEandNS-IRmakequeriesmoreinforma-
compatibility encoder E is initialized from tivethroughLLM-basedexpansion,rewriting,or
C
bge-large-en-v1.5 and trained as a bi-encoder. constraintreasoning,buttheinheritedtopicalem-
|          |     |             |       |                 |     |     | bedding space | still | limits their | ability | to separate |     |
| -------- | --- | ----------- | ----- | --------------- | --- | --- | ------------- | ----- | ------------ | ------- | ----------- | --- |
| Training | has | two stages, | using | WordNet-derived |     |     |               |       |              |         |             |     |
word-level lexical polarity triplets and sentence- satisfyingandviolatingevidenceconsistently.
level triplets from 800 NevIR and 800 ExcluIR DEO and CoDeR reflect a stronger interven-
trainingqueriesexcludedfromtesting. Allqueries, tionbyputtingpressureonconstraint-violatingev-
triplets,andassociateddocumentsusedtotrainE idence itself, treating it as something to be sep-
C
are removed from evaluation splits and reported arated or penalized at the embedding and scor-
test results. We use the same query prefix for ing level rather than only rephrasing the query.
|          |     |            |     |          |         |     | CoDeR makes | this | idea explicit |     | by training | E   |
| -------- | --- | ---------- | --- | -------- | ------- | --- | ----------- | ---- | ------------- | --- | ----------- | --- |
| training | and | inference. | All | reported | results | are |             |      |               |     |             | C   |
averagedover10independentruns,withboththe with satisfying–violating contrasts and applying
method ranking and CoDeR gains remain stable the resulting compatibility signal over topically
V@k
| acrossruns. |     |     |     |     |     |     | plausible | candidates. | Thus, | the | early | and |
| ----------- | --- | --- | --- | --- | --- | --- | --------- | ----------- | ----- | --- | ----- | --- |
FVRgainssuggestthatconstraint-awareretrieval
5.2 GeneralRetrievalQuality
benefitsfromarepresentationwhereconstraintdi-
Although CoDeR targets constraint-sensitive re- rectionisseparablefromtopicalrelatedness,rather
trieval,aninformationretrievalmethodmuststill thanfromqueryrewritingalone.
| preserveordinarytopicalcoverage. |     |                |     |       | Table1there- |        |                           |     |     |     |     |     |
| -------------------------------- | --- | -------------- | --- | ----- | ------------ | ------ | ------------------------- | --- | --- | --- | --- | --- |
|                                  |     |                |     |       |              |        | 5.4 EvaluationonPublished |     |     |     |     |     |
| fore serves                      | as  | a preservation |     | check | rather       | than a |                           |     |     |     |     |     |
claimofgeneral-purposeretrievalsuperiority. Be- Negative-ConstraintBenchmarks
causethesedatasetsmostlyrewardtopicalmatch- We next evaluate CoDeR on published negative-
ing, the strongest dense retrievers can remain constraintretrievalbenchmarks,includingNevIR,
6

CQA
|         |     | SciFact |     | ArguAna |     | FiQA |     | NFCorpus |     |         | SciDocs |
| ------- | --- | ------- | --- | ------- | --- | ---- | --- | -------- | --- | ------- | ------- |
| Methods |     |         |     |         |     |      |     |          |     | Android |         |
nDCG↑ MAP↑ nDCG↑ MAP↑ nDCG↑ MAP↑ nDCG↑ MAP↑ nDCG↑ MAP↑ nDCG↑ MAP↑
BM25 64.82 60.37 34.40 22.56 23.25 17.20 30.98 22.36 39.08 34.32 15.04 8.76
BGE 74.77 69.93 45.99 32.25 45.00 37.15 38.15 28.76 50.22 43.89 22.63 13.80
Contriever 55.04 49.43 33.55 21.90 12.41 8.81 27.12 18.62 30.37 25.56 10.97 5.85
HyDE 67.44 61.98 30.41 19.28 28.51 22.16 31.59 22.46 21.55 17.56 14.21 7.95
NS-IR 74.44 69.92 45.35 32.31 42.68 34.80 36.92 27.66 49.94 43.61 22.59 13.42
DEO 72.09 67.19 70.27 63.08 43.62 35.56 36.68 14.27 44.59 37.91 22.33 13.47
CoDeR-Seq 74.77 69.93 49.36 35.40 45.00 37.15 37.24 28.00 48.02 42.15 21.42 12.86
CoDeR-Union 74.91 70.17 48.78 35.00 44.51 36.80 36.36 27.12 48.17 42.22 21.60 12.96
Table1: GeneralretrievalqualityonBEIR-stylebenchmarks. WereportnDCG@10andMAP@10. Thesedatasets
donotexplicitlytargetconstraintviolationsandareusedtotestwhetherCoDeRpreservestopicalretrievalquality.
|     |     |     | Antonym |     |     |     | Negation |     |     | Exclusion |     |
| --- | --- | --- | ------- | --- | --- | --- | -------- | --- | --- | --------- | --- |
Methods
V@2↓ V@3↓ V@5↓ V@10↓ FVR↑ V@2↓ V@3↓ V@5↓ V@10↓ FVR↑ V@2↓ V@3↓ V@5↓ V@10↓ FVR↑
BM25 90.20 98.04 100.00 100.00 2.11 98.03 100.00 100.00 100.00 2.00 76.47 87.25 97.06 100.00 2.26
BGE 81.37 92.16 99.02 100.00 2.30 90.20 96.08 100.00 100.00 2.02 69.61 82.35 88.24 95.10 2.93
Contriever 78.43 95.10 100.00 100.00 2.21 83.33 92.16 99.02 100.00 2.25 77.45 89.22 96.08 100.00 2.20
HyDE 82.35 92.16 100.00 100.00 2.08 84.31 91.18 99.02 100.00 2.06 66.67 83.33 96.08 99.02 2.48
NS-IR 75.49 92.16 99.02 100.00 2.34 84.31 97.06 100.00 100.00 2.15 68.63 81.37 92.16 99.02 2.72
DEO 73.53 85.29 98.04 100.00 2.42 72.55 86.27 94.12 97.06 2.68 53.85 71.15 79.81 88.46 3.76
CoDeR-Seq 59.80 76.47 91.18 99.02 2.95 50.98 73.53 84.31 92.16 3.53 50.00 60.58 74.04 84.62 4.26
CoDeR-Union 52.94 76.47 93.13 98.04 3.00 49.02 72.55 89.22 90.20 3.47 48.08 61.54 75.96 81.73 4.37
Table2: Constraint-awareretrievalresultsonself-constructedAntonym,Negation,andExclusiondatasets. Metrics
aredefinedinSection5.1.
ExcluIR, and NegConstraint. NevIR focuses on andscored,notsimplyfrombetterpromptwording.
negation-awareretrieval,whileExcluIRevaluates
|                               |     |     |     |                |     |     | 5.5 | AblationStudySummary |     |     |     |
| ----------------------------- | --- | --- | --- | -------------- | --- | --- | --- | -------------------- | --- | --- | --- |
| explicitexclusionconstraints. |     |     |     | ForNevIRandEx- |     |     |     |                      |     |     |     |
cluIR,wemappairedopposite-directionevidence
Wesummarizethecomponentablationshereand
| totheviolatingsetandreportV@k |     |     |     | andFVR.Neg- |     |     |     |     |     |     |     |
| ----------------------------- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
providefullresultsinAppendixG.Theablations
| Constraint | follows |     | a different | released | protocol, |     |      |         |         |            |               |
| ---------- | ------- | --- | ----------- | -------- | --------- | --- | ---- | ------- | ------- | ---------- | ------------- |
|            |         |     |             |          |           |     | test | whether | CoDeR’s | gains come | from the pro- |
so we report its available metrics separately. All posed separation between topical relevance and
| queriesusedtotrainE |       |     | areremovedfromthecor- |            |     |         |                                                 |     |     |     |     |
| ------------------- | ----- | --- | --------------------- | ---------- | --- | ------- | ----------------------------------------------- | --- | --- | --- | --- |
|                     |       |     | C                     |            |     |         | constraintcompatibility,ratherthanfromastronger |     |     |     |     |
| responding          | NevIR |     | and ExcluIR           | evaluation |     | splits, |                                                 |     |     |     |     |
topicalretriever,anotherdenseencoder,orapost-
yieldingheld-outevaluationswithinthepublished
|     |     |     |     |     |     |     | hoc | reranker. | First, changing | the topical | encoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --------------- | ----------- | ------- |
negative-constraintbenchmarks.
hasalargereffectonordinaryretrievalqualitythan
Table 3 shows that the compatibility-scoring on the overall violation-control pattern when the
mechanism remains effective on NevIR and Ex- compatibilityencoderisfixed. Second,replacing
cluIRbeyondourdiagnosticconstruction. Across the compatibility encoder with generic dense en-
both negation and exclusion, topical-matching coders weakens violation-oriented behavior, sug-
or query-rewriting methods still expose viola- gestingsemanticsimilarityalonedoesnotprovide
tions early, while methods that score or penalize thesameconstraint-compatibilitysignal. Third,an
constraint-violatingcandidatesmovethefirstviola- additionalrerankerablationshowsthatastandard
tiondeeper,suggestingthatE capturesacompat- reranker does not consistently delay the first vio-
C
ibilitysignalbeyondasinglediagnostictemplate. latingdocumentwhentopicallysimilarviolations
|     |     |     |     |     |     |     | have | already | entered | the candidate | set. Together, |
| --- | --- | --- | --- | --- | --- | --- | ---- | ------- | ------- | ------------- | -------------- |
Table6reportsresultsonNegConstraintunder
theseresultssupportCoDeR’stwo-signaldesign,
| its released | metrics, |     | since | the full label | structure |     |     |     |     |     |     |
| ------------ | -------- | --- | ----- | -------------- | --------- | --- | --- | --- | --- | --- | --- |
wheretopicalretrievalpreservescoverageandcom-
neededforourviolationdiagnosticsisunavailable.
patibilityscoringcontrolsconstraintdirection.
| The gains | over | NS-IR | provide  | a complementary |          |     |     |                                 |     |     |     |
| --------- | ---- | ----- | -------- | --------------- | -------- | --- | --- | ------------------------------- | --- | --- | --- |
| check:    | even | under | standard | ranking         | metrics, | lo- |     |                                 |     |     |     |
|           |      |       |          |                 |          |     | 5.6 | HyperparameterPolicySensitivity |     |     |     |
calcompatibilityscoringimprovesretrievalunder
negativeconstraints. Togetherwiththeviolation- Webrieflyanalyzepolicysensitivitytoverifythat
orientedresultsabove,thissuggeststhatthebenefit CoDeR is not driven by a single tuned hyperpa-
comesfromhowcandidateevidenceisrepresented rameterpoint;fullheatmapsanddetailedanalysis
7

|     |     |     |     | NevIR |     |     |     |     | ExcluIR |     |     |     |
| --- | --- | --- | --- | ----- | --- | --- | --- | --- | ------- | --- | --- | --- |
Methods
|     |     | V@2↓ | V@3↓ | V@5↓ | V@10↓ | FVR↑ V@2↓ |     | V@3↓ | V@5↓ | V@10↓ |     | FVR↑ |
| --- | --- | ---- | ---- | ---- | ----- | --------- | --- | ---- | ---- | ----- | --- | ---- |
BM25 43.00 65.20 85.00 90.40 3.56 90.20 93.90 96.60 97.70 1.94
BGE 36.80 60.30 85.60 90.30 3.70 96.10 97.70 98.20 99.10 1.75
Contriever 42.10 64.40 84.20 88.50 3.69 91.60 93.90 96.20 98.00 1.91
HyDE 41.30 63.40 86.10 90.00 3.58 57.40 66.00 75.00 81.80 4.21
NS-IR 35.80 60.90 84.80 91.40 3.71 95.60 97.10 98.10 98.70 1.79
DEO 31.10 56.50 81.90 88.60 4.02 51.10 59.60 66.10 74.30 4.93
CoDeR-Seq 28.50 52.90 79.40 86.20 4.26 47.70 56.60 64.40 73.70 5.06
CoDeR-Union 30.20 53.20 78.80 86.00 4.28 45.00 57.20 65.30 65.70 5.33
Table3: ResultsonpublishedNevIRandExcluIRbenchmarks. MetricsaredefinedinSection5.1.
areprovidedinAppendixF.AcrossNegationand Method Time(h)↓ Cost($)↓ Tokens(K)↓
| ExcluIR, | the main | trade-off |     | is between | preserv- |      |     |       |     |       |        |     |
| -------- | -------- | --------- | --- | ---------- | -------- | ---- | --- | ----- | --- | ----- | ------ | --- |
|          |          |           |     |            |          | HyDE |     | 46.67 |     | 32.30 | 3854.0 |     |
ingsatisfyingevidence,measuredbyRecall@10, NS-IR 53.76 32.22 4531.4
|              |     |       |           |           |      | DEO |     | 9.64 |     | 5.12 | 1376.9 |     |
| ------------ | --- | ----- | --------- | --------- | ---- | --- | --- | ---- | --- | ---- | ------ | --- |
| and delaying | the | first | violating | document, | mea- |     |     |      |     |      |        |     |
sured by FVR. CoDeR-Seq is more sensitive to CoDeR-Seq 0.08 0.00 0.0
|                       |     |     |                       |     |     | CoDeR-Union |     | 0.07 |     | 0.00 | 0.0 |     |
| --------------------- | --- | --- | --------------------- | --- | --- | ----------- | --- | ---- | --- | ---- | --- | --- |
| theabsolutethresholdτ |     |     | becauseitdependsonraw |     |     |             |     |      |     |      |     |     |
compatibility-score calibration, whereas CoDeR- Table 4: Aggregate inference efficiency across six
Unionusesarelativefilterpercentile,γ q ,andshows constraint-awareretrievalsettings. Timereportstotal
smootherbehavioracrossdatasets. Theheatmaps wall-clockhoursincludingofflinepreparationandon-
|     |     |     |     |     |     | linequeryexecution. |     | Costandtokensreportexternal |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------- | --- | --------------------------- | --- | --- | --- | --- |
confirmthatαgovernsthetrade-offbetweentopi-
|     |     |     |     |     |     | APIusage. | Lowerisbetter. |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --------- | -------------- | --- | --- | --- | --- | --- |
calcoverageandcompatibility-drivenranking,and
| that filtering | strength |     | adjusts | constraint | enforce- |     |     |     |     |     |     |     |
| -------------- | -------- | --- | ------- | ---------- | -------- | --- | --- | --- | --- | --- | --- | --- |
mentacrossacontinuousrangeratherthancollaps-
|     |     |     |     |     |     | ingintoE | andavoidsrepeatedLLMcallsatin- |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | -------- | ------------------------------ | --- | --- | --- | --- | --- |
C
ingtoasingletunedoperatingpoint. ferencetime,makingitattractiveforcost-sensitive
andprivacy-sensitiveretrievaldeployments.
5.7 EfficiencyAnalysis
6 Conclusion
Wereportinference-timeefficiencyseparatelybe-
| cause time,        | external  | API          | cost, | and          | token usage |                              |             |          |           |                   |             |     |
| ------------------ | --------- | ------------ | ----- | ------------ | ----------- | ---------------------------- | ----------- | -------- | --------- | ----------------- | ----------- | --- |
|                    |           |              |       |              |             | We presented                 |             | CoDeR,   |           | a local           | constraint- |     |
| reflect deployment |           | overhead     |       | rather than  | retrieval   |                              |             |          |           |                   |             |     |
|                    |           |              |       |              |             | compatible                   | information |          | retrieval |                   | method      | for |
| accuracy.          | Table     | 4 aggregates |       | measurements | over        |                              |             |          |           |                   |             |     |
|                    |           |              |       |              |             | constraint-sensitivequeries. |             |          |           | CoDeRtargetscases |             |     |
| Antonym,           | Negation, | Exclusion,   |       | NevIR,       | ExcluIR,    |                              |             |          |           |                   |             |     |
|                    |           |              |       |              |             | where topically              |             | relevant | documents |                   | support     | the |
andNegConstraint,withfullper-settingmeasure-
|     |     |     |     |     |     | wrong constraint |     | direction, |     | such as | negation, | ex- |
| --- | --- | --- | --- | --- | --- | ---------------- | --- | ---------- | --- | ------- | --------- | --- |
mentsinAppendixD.Thereportedtimeincludes
|     |     |     |     |     |     | clusion,orantonymicpreferences. |     |     |     |     | Bycombining |     |
| --- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- | ----------- | --- |
offlinepreparationsuchasmodelloading,corpus
|     |     |     |     |     |     | topical relevance |     | with | a constraint |     | compatibility |     |
| --- | --- | --- | --- | --- | --- | ----------------- | --- | ---- | ------------ | --- | ------------- | --- |
encoding,indexconstruction,andcacheinitializa-
scorer,CoDeRreducesearlyexposuretoconstraint-
tion,plusonlineprocessingoverallqueries.
|       |      |         |       |                     |     | violating                  | evidence | on  | controlled | diagnostics |     | and   |
| ----- | ---- | ------- | ----- | ------------------- | --- | -------------------------- | -------- | --- | ---------- | ----------- | --- | ----- |
| CoDeR | runs | locally | after | corpus pre-encoding |     |                            |          |     |            |             |     |       |
|       |      |         |       |                     |     | public negative-constraint |          |     |            | benchmarks, |     | while |
whereinferenceonlyrequiresqueryencoding,vec-
|              |           |        |             |                |          | largely preserving |     | ordinary                  |     | retrieval | quality.  | Our |
| ------------ | --------- | ------ | ----------- | -------------- | -------- | ------------------ | --- | ------------------------- | --- | --------- | --------- | --- |
| tor scoring, | candidate |        | filtering,  | and reranking, | so       |                    |     |                           |     |           |           |     |
|              |           |        |             |                |          | results suggest    |     | that constraint-sensitive |     |           | retrieval |     |
| it incurs    | no API    | tokens | or external | API            | cost. By |                    |     |                           |     |           |           |     |
shouldbeevaluatednotonlybytopicalrelevance,
contrast,HyDE,NS-IR,andDEOrepeatedlycall
butalsobywhetherearly-rankeddocumentssatisfy
LLMAPIsforhypotheticaldocumentgeneration,
thedirectionoftheuser’sstatedconstraint.
| logical translation, |     | or  | query | decomposition. | For |     |     |     |     |     |     |     |
| -------------------- | --- | --- | ----- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
fairness,thoseAPI-basedbaselinesareevaluated
Limitations
| with GPT-4o | where | applicable |     | and their | logged |     |     |     |     |     |     |     |
| ----------- | ----- | ---------- | --- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- |
tokensandcostsareincluded. CoDeRfocusesonexplicitnegativeconstraintsex-
Theefficiencyresultshighlightthedifferencebe- pressedthroughantonymy,negation,andexclusion.
tweenprompt/API-basedandrepresentation-based Other constraint types, such as numeric, tempo-
interventions. CoDeRamortizesconstraintlearn- ral,multi-hop,implicit,orpragmaticallyambigu-
8

ous constraints, may require additional supervi- Taegyeong Lee, Jiwon Park, Seunghyun Hwang, and
sionordifferentcompatibilitysignals. CoDeRis JooYoung Jang. 2026. Deo: Training-free direct
embeddingoptimizationfornegation-awareretrieval.
alsotrainedwithEnglishlexical-polarityresources
arXivpreprintarXiv:2603.09185.
andsentence-leveltripletsfromnegative-constraint
benchmarks. Although all training queries and PatrickLewis,EthanPerez,AleksandraPiktus,Fabio
associateddocumentsareremovedfromthecorre- Petroni,VladimirKarpukhin,NamanGoyal,Hein-
|     |     |     |     |     |     | richKüttler, | MikeLewis, |     | Wen-tauYih, |     | TimRock- |     |
| --- | --- | --- | --- | --- | --- | ------------ | ---------- | --- | ----------- | --- | -------- | --- |
spondingevaluationsplits,theNevIRandExcluIR
|     |     |     |     |     |     | täschel,and1others.2020. |     |     | Retrieval-augmentedgen- |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | ----------------------- | --- | --- | --- |
resultsarebestviewedasheld-outin-domaineval- erationforknowledge-intensivenlptasks. Advances
uations rather than fully out-of-domain transfer. inneuralinformationprocessingsystems,33:9459–
9474.
ApplyingCoDeRtonewlanguagesorspecialized
domainsmayrequireconstructingpolaritytriplets NelsonFLiu,KevinLin,JohnHewitt,AshwinParan-
andadaptingthecompatibilityencoder.
jape,MicheleBevilacqua,FabioPetroni,andPercy
Our main evaluation is retrieval-side. CoDeR Liang.2024. Lostinthemiddle: Howlanguagemod-
|         |        |       |          |     |             | elsuselongcontexts. |     | Transactionsoftheassociation |     |     |     |     |
| ------- | ------ | ----- | -------- | --- | ----------- | ------------------- | --- | ---------------------------- | --- | --- | --- | --- |
| aims to | reduce | early | exposure | to  | constraint- |                     |     |                              |     |     |     |     |
forcomputationallinguistics,12:157–173.
violatingdocumentsintherankedlist,butitdoes
notbyitselfguaranteedownstreamanswerfactu- GeorgeAMiller.1995. Wordnet: alexicaldatabasefor
|                  |     |           |            |     |          | english. | CommunicationsoftheACM,38(11):39–41. |     |     |     |     |     |
| ---------------- | --- | --------- | ---------- | --- | -------- | -------- | ------------------------------------ | --- | --- | --- | --- | --- |
| ality or safety. |     | The small | downstream |     | probe is |          |                                      |     |     |     |     |     |
includedonlyasapreliminarydownstreamvalida- AaronvandenOord,YazheLi,andOriolVinyals.2018.
tion. Evaluatinghowcompatibility-awareretrieval Representationlearningwithcontrastivepredictive
composeswithdownstreamreaders,rerankers,and coding. arXivpreprintarXiv:1807.03748.
generatorsremainsfuturework.
YingqiQu,YuchenDing,JingLiu,KaiLiu,Ruiyang
Potential risks arise if compatibility scores are Ren,WayneXinZhao,DaxiangDong,HuaWu,and
miscalibrated. CoDeRmayincorrectlydemoterel- HaifengWang.2021. Rocketqa: Anoptimizedtrain-
|     |     |     |     |     |     | ing approach | to  | dense | passage | retrieval | for | open- |
| --- | --- | --- | --- | --- | --- | ------------ | --- | ----- | ------- | --------- | --- | ----- |
evantdocumentswhenconstraintsareimplicit,am-
|     |     |     |     |     |     | domainquestionanswering. |     |     |     | InProceedingsofthe |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | ------------------ | --- | --- |
biguous,orexpresseddifferentlyfromthetraining
2021conferenceoftheNorthAmericanchapterof
patterns. Inhigh-stakessettingssuchasmedicalor theassociationforcomputationallinguistics: human
legalretrieval,CoDeRshouldthereforebeusedas languagetechnologies,pages5835–5847.
aretrieval-sideaidratherthanastandalonesafety
|     |     |     |     |     |     | NilsReimersandIrynaGurevych.2019. |     |     |     |     | Sentence-bert: |     |
| --- | --- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | --- | -------------- | --- |
mechanism, with downstream verification or hu- Sentenceembeddingsusingsiamesebert-networks.
manreviewwhenappropriate. InProceedingsofthe2019conferenceonempirical
methodsinnaturallanguageprocessingandthe9th
internationaljointconferenceonnaturallanguage
processing(EMNLP-IJCNLP),pages3982–3992.
References
|                     |     |           |            |       |           | StephenRobertsonandHugoZaragoza.2009. |                   |            |     |      | Theprob- |         |
| ------------------- | --- | --------- | ---------- | ----- | --------- | ------------------------------------- | ----------------- | ---------- | --- | ---- | -------- | ------- |
| Hanjun Cho          | and | Jay-Yoon  | Lee.       | 2026. | Rare:     |                                       |                   |            |     |      |          |         |
|                     |     |           |            |       |           | abilistic                             | relevance         | framework: |     | BM25 | and      | beyond, |
| Redundancy-aware    |     | retrieval | evaluation |       | framework |                                       |                   |            |     |      |          |         |
|                     |     |           |            |       |           | volume4.                              | NowPublishersInc. |            |     |      |          |         |
| for high-similarity |     | corpora.  |            | arXiv | preprint  |                                       |                   |            |     |      |          |         |
arXiv:2604.19047.
|     |     |     |     |     |     | Nandan Thakur,                          |     | Nils Reimers, |     | Andreas | Rücklé, | Ab-   |
| --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | ------------- | --- | ------- | ------- | ----- |
|     |     |     |     |     |     | hishekSrivastava,andIrynaGurevych.2021. |     |               |     |         |         | Beir: |
LuyuGao,XueguangMa,JimmyLin,andJamieCallan.
|     |     |     |     |     |     | A heterogenous |     | benchmark |     | for zero-shot |     | evalua- |
| --- | --- | --- | --- | --- | --- | -------------- | --- | --------- | --- | ------------- | --- | ------- |
2023. Precisezero-shotdenseretrievalwithoutrel-
|        |         |                |     |        |             | tionofinformationretrievalmodels. |     |     |     |     | arXivpreprint |     |
| ------ | ------- | -------------- | --- | ------ | ----------- | --------------------------------- | --- | --- | --- | --- | ------------- | --- |
| evance | labels. | In Proceedings |     | of the | 61st Annual |                                   |     |     |     |     |               |     |
arXiv:2104.08663.
| Meeting | of the | Association | for | Computational | Lin- |     |     |     |     |     |     |     |
| ------- | ------ | ----------- | --- | ------------- | ---- | --- | --- | --- | --- | --- | --- | --- |
guistics(Volume1: LongPapers),pages1762–1777. NavveWasserman,OliverHeinimann,YuvalGolbari,
TalZimbalist,EliSchwartz,andMichalIrani.2025.
GautierIzacard,MathildeCaron,LucasHosseini,Se- Docrerank: Single-pagehardnegativequerygenera-
| bastian | Riedel, | Piotr Bojanowski, |     | Armand | Joulin, |     |     |     |     |     |     | InPro- |
| ------- | ------- | ----------------- | --- | ------ | ------- | --- | --- | --- | --- | --- | --- | ------ |
tionfortrainingmulti-modalragrerankers.
| andEdouardGrave.2021. |     |     | Unsuperviseddensein- |     |     |     |     |     |     |     |     |     |
| --------------------- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ceedingsofthe2025ConferenceonEmpiricalMeth-
arXiv
formationretrievalwithcontrastivelearning. ods in Natural Language Processing, pages 8651–
| preprintarXiv:2112.09118. |     |     |     |     |     | 8669. |     |     |     |     |     |     |
| ------------------------- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
VladimirKarpukhin,BarlasOguz,SewonMin,Patrick OrionWeller,DawnLawrie,andBenjaminVanDurme.
Lewis,LedellWu,SergeyEdunov,DanqiChen,and 2024. Nevir:Negationinneuralinformationretrieval.
Wen-tauYih.2020. Densepassageretrievalforopen- InProceedingsofthe18thConferenceoftheEuro-
domainquestionanswering. InProceedingsofthe peanChapteroftheAssociationforComputational
2020 conference on empirical methods in natural Linguistics (Volume 1: Long Papers), pages 2274–
| languageprocessing(EMNLP),pages6769–6781. |     |     |     |     |     | 2287. |     |     |     |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
9

ShitaoXiao,ZhengLiu,PeitianZhang,NiklasMuen- Eachqueryisassignedthreetypesofevidence
nighoff,DefuLian,andJian-YunNie.2024. C-pack: labels. Aconstraint-satisfyingdocumentmatches
| Packedresourcesforgeneralchineseembeddings. |     |        |                    |     |     | In    |           |       |     |           |     |        |             |
| ------------------------------------------- | --- | ------ | ------------------ | --- | --- | ----- | --------- | ----- | --- | --------- | --- | ------ | ----------- |
|                                             |     |        |                    |     |     |       | the query | topic | and | satisfies | the | stated | constraint. |
| Proceedings                                 |     | of the | 47th international |     | ACM | SIGIR |           |       |     |           |     |        |             |
Aconstraint-violatingdocumentmatchesthetopic
conferenceonresearchanddevelopmentininforma-
|     |     |     |     |     |     |     | but supports |     | the opposite |     | constraint | direction. | A   |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------------ | --- | ---------- | ---------- | --- |
tionretrieval,pages641–649.
|     |     |     |     |     |     |     | neutral | topical | document |     | is related | to  | the query |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------- | -------- | --- | ---------- | --- | --------- |
LeeXiong,ChenyanXiong,YeLi,Kwok-FungTang,
topicbutdoesnotexplicitlysatisfyorviolatethe
JialinLiu,PaulBennett,JunaidAhmed,andArnold
Overwijk.2020. Approximatenearestneighborneg- constraint. For nDCG computation, we assign
ative contrastive learning for dense text retrieval. graded relevance scores of 2 to satisfying docu-
arXivpreprintarXiv:2007.00808.
ments,1toneutraltopicaldocuments,and0tovio-
GanlinXu,ZhoujiaZhang,WangyiMei,JiaqingLiang, latingdocuments. Forviolation-orientedmetrics,
WeijiaLu,XiaodongZhang,ZhifeiYang,Xiaofeng onlyconstraint-violatingdocumentsarecountedas
| Ma,YanghuaXiao,andDeqingYang.2025. |     |     |     |     | Logical |     |     |     |     |     |     |     |     |
| ---------------------------------- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
violations.
| consistencyisvital:                        |     |     | Neural-symbolicinformationre- |     |              |     |                                  |     |     |     |     |     |     |
| ------------------------------------------ | --- | --- | ----------------------------- | --- | ------------ | --- | -------------------------------- | --- | --- | --- | --- | --- | --- |
| trievalfornegative-constraintqueries.      |     |     |                               |     | InFindingsof |     |                                  |     |     |     |     |     |     |
| theAssociationforComputationalLinguistics: |     |     |                               |     |              | ACL |                                  |     |     |     |     |     |     |
|                                            |     |     |                               |     |              |     | B ConstraintCompatibilityEncoder |     |     |     |     |     |     |
2025,pages1828–1847.
TrainingDetails
WenhaoZhang,MengqiZhang,ShiguangWu,Jiahuan
Pei,ZhaochunRen,MaartenDeRijke,ZhuminChen,
|                     |     |     |          |                    |     |     | TheconstraintcompatibilityencoderE |     |     |     |     |     | isinitial- |
| ------------------- | --- | --- | -------- | ------------------ | --- | --- | ---------------------------------- | --- | --- | --- | --- | --- | ---------- |
| andPengjieRen.2025. |     |     | Excluir: | Exclusionaryneural |     |     |                                    |     |     |     |     |     | C          |
information retrieval. In Proceedings of the AAAI izedfrombge-large-en-v1.5andtrainedasabi-
| Conference        |     | on Artificial | Intelligence, |     | volume | 39, |                         |     |     |                       |                 |     |     |
| ----------------- | --- | ------------- | ------------- | --- | ------ | --- | ----------------------- | --- | --- | --------------------- | --------------- | --- | --- |
|                   |     |               |               |     |        |     | encoderwithmeanpooling. |     |     |                       | Wesetthemaximum |     |     |
| pages13295–13303. |     |               |               |     |        |     | sequencelengthto128.    |     |     | Weuseatwo-stagetrain- |                 |     |     |
A AdditionalDatasetConstruction ingrecipe. Inthefirststage,wetrainonWordNet-
| Details      |         |       |             |     |             |     | derivedword-levelpolaritytriplets,wherethepos- |            |          |      |        |          |          |
| ------------ | ------- | ----- | ----------- | --- | ----------- | --- | ---------------------------------------------- | ---------- | -------- | ---- | ------ | -------- | -------- |
|              |         |       |             |     |             |     | itive and                                      | negative   | examples |      | differ | by       | antonymy |
| We construct |         | three | diagnostic  |     | test sets   | to  |                                                |            |          |      |        |          |          |
|              |         |       |             |     |             |     | or polarity                                    | direction. |          | This | stage  | provides | a basic  |
| evaluate     | whether |       | a retriever | can | distinguish |     |                                                |            |          |      |        |          |          |
lexicalpolaritysignalbeforesentence-leveltrain-
constraint-satisfyingevidencefromtopicallysim-
ing(Miller,1995).
| ilar constraint-violating |      |          | evidence. |          | The Antonym |      |        |        |        |     |          |     |             |
| ------------------------- | ---- | -------- | --------- | -------- | ----------- | ---- | ------ | ------ | ------ | --- | -------- | --- | ----------- |
|                           |      |          |           |          |             |      | In the | second | stage, | we  | continue |     | training on |
| set is built              | from | SciFact, | the       | Negation | set         | from |        |        |        |     |          |     |             |
sentence-leveltripletsconstructedfromNevIRand
| SciDocs, | and | the Exclusion |     | set from | NFCorpus. |     |          |     |        |     |          |         |      |
| -------- | --- | ------------- | --- | -------- | --------- | --- | -------- | --- | ------ | --- | -------- | ------- | ---- |
|          |     |               |     |          |           |     | ExcluIR. | We  | sample | 800 | training | queries | from |
Table5summarizestheresultingdatasetsizes.
|     |     |     |     |     |     |     | NevIR | and 800 | training |     | queries | from | ExcluIR. |
| --- | --- | --- | --- | --- | --- | --- | ----- | ------- | -------- | --- | ------- | ---- | -------- |
(q,d+,d−),
Dataset Source Queries Documents Each training instance is a triplet
|     |     |     |     |     |     |     | d+  |     |     |     |     |     | d−  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Antonym SciFact 102 306 where satisfies the query constraint and is
Negation SciDocs 104 304 topically related but constraint-violating. These
| Exclusion |     | NFCorpus | 104 |     | 312 |     |                |     |          |       |     |          |         |
| --------- | --- | -------- | --- | --- | --- | --- | -------------- | --- | -------- | ----- | --- | -------- | ------- |
|           |     |          |     |     |     |     | sentence-level |     | triplets | teach | E   | to score | compat- |
C
|          |            |     |                      |     |            |     | ibility at | the passage |     | level | rather | than | only at the |
| -------- | ---------- | --- | -------------------- | --- | ---------- | --- | ---------- | ----------- | --- | ----- | ------ | ---- | ----------- |
| Table 5: | Statistics | of  | the self-constructed |     | diagnostic |     |            |             |     |       |        |      |             |
wordlevel.
datasets.
|     |     |     |     |     |     |     | To avoid | data | leakage, |     | all queries |     | and triples |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---- | -------- | --- | ----------- | --- | ----------- |
For each source corpus, we first select candi- usedtotrainE areremovedfromthecorrespond-
C
datedocumentswithsufficientlengthandsegment ing evaluation splits. They do not appear in any
themintoshortersub-documents. FortheAntonym reported NevIR or ExcluIR test results. We use
set, we identify a keyword with a valid antonym the same BGE-style query prefix during training
andconstructpairedqueriesandpaireddocuments andinference. TheNevIRandExcluIRresultsare
byreplacingthekeywordwithitsantonymwhile held-out in-domain evaluations rather than fully
keepingthesurroundingtopicfixed. FortheNega- out-of-domain transfer. The model is optimized
tionset,weconstructpairedexamplesbychanging withthemultiplenegativesrankinglossdescribed
whetheraclaimorrelationisassertedornegated. in Section 4.2, using the explicit violating docu-
For the Exclusion set, we construct queries that ment and in-batch negatives as contrastive nega-
requireexcludingaspecificattribute,entity,orcon- tives. Wetrainforoneepochwithabatchsizeof
dition,togetherwithdocumentsthateithersatisfy 16,usingAdamWwithalearningrateof2×10−5
| orviolatetheexclusion. |     |     |     |     |     |     | throughoutbothstages. |     |     |     |     |     |     |
| ---------------------- | --- | --- | --- | --- | --- | --- | --------------------- | --- | --- | --- | --- | --- | --- |
10

C NegConstraintExperimentResult
|             |     |           |     |         |         | Setting  | Method      |     | Time(s)↓ | Cost($)↓ | Tokens(K)↓ |       |
| ----------- | --- | --------- | --- | ------- | ------- | -------- | ----------- | --- | -------- | -------- | ---------- | ----- |
|             |     |           |     |         |         |          | HyDE        |     | 3900.98  |          | 1.22       | 146.3 |
|             |     |           |     |         |         |          | NS-IR       |     | 7190.02  |          | 0.85       | 128.8 |
|             |     |           |     |         |         | Antonym  | DEO         |     | 180.94   |          | 0.18       | 50.5  |
| Methods     |     | Recall@5↑ |     | nDCG@5↑ | MAP@10↑ |          |             |     |          |          |            |       |
|             |     |           |     |         |         |          | CoDeR-Seq   |     |          | 6.14     | 0.00       | 0.0   |
|             |     |           |     |         |         |          | CoDeR-Union |     |          | 6.19     | 0.00       | 0.0   |
| NS-IR       |     | 95.96     |     | 76.64   | 70.05   |          |             |     |          |          |            |       |
|             |     |           |     |         |         |          | HyDE        |     | 3222.89  |          | 1.24       | 148.8 |
| CoDeR-Seq   |     | 96.46     |     | 81.20   | 76.00   |          | NS-IR       |     | 6979.14  |          | 0.98       | 144.0 |
| CoDeR-Union |     | 96.97     |     | 81.61   | 76.38   | Negation | DEO         |     | 187.12   |          | 0.18       | 50.9  |
|             |     |           |     |         |         |          | CoDeR-Seq   |     |          | 6.44     | 0.00       | 0.0   |
|             |     |           |     |         |         |          | CoDeR-Union |     |          | 6.81     | 0.00       | 0.0   |
Table6: ResultsonthereleasedNegConstraintbench-
|     |     |     |     |     |     |     | HyDE |     | 4232.36 |     | 1.00 | 124.1 |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | ------- | --- | ---- | ----- |
mark. Followingthereleasedevaluationprotocol,we
|     |     |     |     |     |     |     | NS-IR |     | 8249.47 |     | 1.27 | 177.9 |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | ------- | --- | ---- | ----- |
report Recall@5, nDCG@5, and MAP@10. Higher Exclusion DEO 190.24 0.19 51.9
| valuesarebetter. |     |     |     |     |     |       | CoDeR-Seq   |     |          | 7.36 | 0.00 | 0.0    |
| ---------------- | --- | --- | --- | --- | --- | ----- | ----------- | --- | -------- | ---- | ---- | ------ |
|                  |     |     |     |     |     |       | CoDeR-Union |     |          | 7.72 | 0.00 | 0.0    |
|                  |     |     |     |     |     |       | HyDE        |     | 56204.29 |      | 9.81 | 1183.0 |
|                  |     |     |     |     |     |       | NS-IR       |     | 61621.02 |      | 5.18 | 755.3  |
|                  |     |     |     |     |     | NevIR | DEO         |     | 16516.02 |      | 1.70 | 487.7  |
D FullEfficiencyAccounting
|     |     |     |     |     |     |     | CoDeR-Seq   |     | 131.23 |     | 0.00 | 0.0 |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------ | --- | ---- | --- |
|     |     |     |     |     |     |     | CoDeR-Union |     | 65.95  |     | 0.00 | 0.0 |
Table7reportstheper-settingefficiencymeasure-
|     |     |     |     |     |     |     | HyDE |     | 69233.49 | 13.85 |     | 1668.9 |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | -------- | ----- | --- | ------ |
ments used in Section 5.7. Time is reported in NS-IR 69775.07 10.65 1501.5
|     |     |     |     |     |     | ExcluIR | DEO |     | 17620.81 |     | 2.03 | 531.4 |
| --- | --- | --- | --- | --- | --- | ------- | --- | --- | -------- | --- | ---- | ----- |
seconds,APIcostinUSdollars,andtokenusage
|              |     |     |     |     |     |     | CoDeR-Seq   |     | 66.70    |       | 0.00 | 0.0    |
| ------------ | --- | --- | --- | --- | --- | --- | ----------- | --- | -------- | ----- | ---- | ------ |
| inthousands. |     |     |     |     |     |     | CoDeR-Union |     | 89.09    |       | 0.00 | 0.0    |
|              |     |     |     |     |     |     | HyDE        |     | 31213.27 |       | 5.18 | 582.9  |
|              |     |     |     |     |     |     | NS-IR       |     | 39738.51 | 13.28 |      | 1823.8 |
E ViolationSurvivalAnalysis NegConstraint DEO 4782.49 0.83 204.4
|     |     |     |     |     |     |     | CoDeR-Seq |     | 82.37 |     | 0.00 | 0.0 |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | ----- | --- | ---- | --- |
Weproviderank-wiseviolationsurvivalcurvesas CoDeR-Union 86.74 0.00 0.0
|     |     |     |     |     |     |     | HyDE |     | 716.90 |     | 0.31 | 35.8 |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | ------ | --- | ---- | ---- |
acomplementaryviewofthediagnosticresultsin
|     |     |     |     |     |     |     | NS-IR |     | 732.22 |     | 0.13 | 19.8 |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | ------ | --- | ---- | ---- |
Table 2. For each query, the first violating rank End-to-End DEO 77.73 0.04 10.4
|     |     |     |     |     |     |     | CoDeR-Seq |     |     | 1.58 | 0.00 | 0.0 |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | ---- | ---- | --- |
records the earliest retrieved position at which a CoDeR-Union 0.94 0.00 0.0
| constraint-violating |     |     | document | appears. | The sur- |         |                  |     |            |             |     |      |
| -------------------- | --- | --- | -------- | -------- | -------- | ------- | ---------------- | --- | ---------- | ----------- | --- | ---- |
|                      |     |     |          |          |          | Table7: | Full per-setting |     | efficiency | accounting. |     | Time |
vivalvalueatrankkisthefractionofquerieswhose
|     |     |     |     |     |     | is reported | in seconds. |     | Cost is | estimated | API | cost in |
| --- | --- | --- | --- | --- | --- | ----------- | ----------- | --- | ------- | --------- | --- | ------- |
firstviolatingrankisgreaterthank,i.e.,thequery
|     |     |     |     |     |     | US dollars. | Tokens | are | API tokens | consumed |     | during |
| --- | --- | --- | --- | --- | --- | ----------- | ------ | --- | ---------- | -------- | --- | ------ |
hasnotexposedviolatingevidencewithinthefirst
|             |            |     |        |     |                  | inference,reportedinthousands. |     |     |     | TheEnd-to-Endset- |     |     |
| ----------- | ---------- | --- | ------ | --- | ---------------- | ------------------------------ | --- | --- | --- | ----------------- | --- | --- |
| k retrieved | documents. |     | Higher |     | curves therefore |                                |     |     |     |                   |     |     |
tingisincludedforcompletenessbutisnotpartofthe
indicatesaferearlyretrievedcontexts. aggregateefficiencytableinSection5.7.
| Figure | 4 shows | that | CoDeR-Seq |     | and CoDeR- |     |     |     |     |     |     |     |
| ------ | ------- | ---- | --------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
Uniongenerallymaintainhigherno-violationrates
atearlyranksthantopicalorquery-rewritingbase- theretrievedlist,whileFVRmeasureshowfarthe
lines. ThisindicatesthatCoDeRchangesthetim- firstconstraint-violatingdocumentispusheddown
ingofviolationexposureratherthanonlyimprov-
|     |     |     |     |     |     | the ranking. | For | the | FVR heatmaps, |     | each | cell re- |
| --- | --- | --- | --- | --- | --- | ------------ | --- | --- | ------------- | --- | ---- | -------- |
ing an aggregate FVR value. By delaying the ports the raw FVR normalized by k +1 with
max
firstconstraint-violatingdocument,compatibility- k = 10,soavalueof1indicatesthatnoviola-
max
awareretrievalreducesthechancethatdownstream tionisfoundwithinthetop10. ForCoDeR-Seq,the
systemsconsumingonlythetopfewretrieveddoc-
gridvariestheinterpolationweightαandtheabso-
umentsencounterdirectionallywrongevidence. lutecompatibilitythresholdτ. ForCoDeR-Union,
|     |     |     |     |     |     | the grid | varies | α and | the relative | filter | percentile. |     |
| --- | --- | --- | --- | --- | --- | -------- | ------ | ----- | ------------ | ------ | ----------- | --- |
F HyperparameterPolicySensitivity
|               |     |               |                    |             |               | In the six                                | BEIR-style |           | datasets, | CoDeR-Seq   |     | uses    |
| ------------- | --- | ------------- | ------------------ | ----------- | ------------- | ----------------------------------------- | ---------- | --------- | --------- | ----------- | --- | ------- |
|               |     |               |                    |             |               | α = 0.5                                   | and τ      | = 0.9,    | while     | CoDeR-Union |     | uses    |
| We provide    | the | full          | policy-sensitivity |             | heatmaps      |                                           |            |           |           |             |     |         |
|               |     |               |                    |             |               | α = 0.8andarelativefilterpercentileof0.8. |            |           |           |             |     | For     |
| for CoDeR-Seq |     | and           | CoDeR-Union.       |             | Figures 5     |                                           |            |           |           |             |     |         |
|               |     |               |                    |             |               | the other                                 | datasets,  | CoDeR-Seq |           | uses        | α = | 0.2 and |
| and 6 report  |     | the Recall@10 |                    | sensitivity | patterns      |                                           |            |           |           |             |     |         |
|               |     |               |                    |             |               | τ = 0.3,whileCoDeR-Unionusesα             |            |           |           |             | =   | 0.3anda |
| for CoDeR-Seq |     | and           | CoDeR-Union,       |             | respectively, |                                           |            |           |           |             |     |         |
relativefilterpercentileof0.1.
| while Figures |     | 7 and | 8 report | the | corresponding |     |     |     |     |     |     |     |
| ------------- | --- | ----- | -------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
FVRsensitivitypatterns. TheanalysisusesNega- Together, Figures5–8showthatCoDeRisnot
tion and ExcluIR as representative datasets and drivenbyasingleaccidentalsetting. CoDeR-Seq
reports two complementary metrics. Recall@10 haseffectivebandsoverαandτ,butbecauseτ is
measureswhethersatisfyingevidenceremainsin anabsolutecompatibilitythreshold,theusefulre-
11

|     |     |     |         | BGE | NS-IR | DEO      | CoDeR-Seq | CoDeR-Union |     |     |           |     |     |
| --- | --- | --- | ------- | --- | ----- | -------- | --------- | ----------- | --- | --- | --------- | --- | --- |
|     |     |     | Antonym |     |       | Negation |           |             |     |     | Exclusion |     |     |
1.0
k knar ot pu etar noitaloiv-oN 0.8
0.6
0.4
0.2
0.0
1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10
|     |     |     | Rank k |     |     |     | Rank k |     |     |     | Rank k |     |     |
| --- | --- | --- | ------ | --- | --- | --- | ------ | --- | --- | --- | ------ | --- | --- |
Figure4:ViolationsurvivalcurvesonthediagnosticAntonym,Negation,andExclusiondatasets. They-axisreports
thefractionofquerieswithnoconstraint-violatingevidenceuptorankk. Highercurvesindicatethatviolationsare
delayeddeeperintherankedlist. CoDeRvariantskeephigherno-violationratesatearlyranks,complementingthe
V@kandFVRresultsinTable2.
|     | Negation |     |     | ExcluIR |     |                              |                   | Negation |     |                   |     | ExcluIR |                              |
| --- | -------- | --- | --- | ------- | --- | ---------------------------- | ----------------- | -------- | --- | ----------------- | --- | ------- | ---------------------------- |
| 1.0 |          |     |     |         |     |                              | 1.0               |          |     |                   |     |         |                              |
|     |          |     |     |         |     | 0.975                        |                   |          |     |                   |     |         | 0.975                        |
| 0.8 |          |     |     |         |     | 0.950                        | 0.8               |          |     |                   |     |         | 0.950                        |
|     |          |     |     |         |     | )retteb si rehgih( 01@llaceR |                   |          |     |                   |     |         | )retteb si rehgih( 01@llaceR |
|     |          |     |     |         |     | 0.925                        |                   |          |     |                   |     |         | 0.925                        |
| 0.6 |          |     |     |         |     | 0.900                        | elitnecrep retlif |          |     | elitnecrep retlif |     |         | 0.900                        |
0.6
| uat |     |     | uat |     |     | 0.875 |     |     |     |     |     |     | 0.875 |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | ----- |
| 0.4 |     |     |     |     |     | 0.850 |     |     |     |     |     |     | 0.850 |
|     |     |     |     |     |     | 0.825 | 0.4 |     |     |     |     |     | 0.825 |
| 0.2 |     |     |     |     |     | 0.800 |     |     |     |     |     |     | 0.800 |
|     |     |     |     |     |     | 0.775 | 0.2 |     |     |     |     |     | 0.775 |
| 0.0 |     |     |     |     |     |       | 0.1 |     |     |     |     |     |       |
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
|     | alpha |     |     | alpha |     |     |     | alpha |     |     |     | alpha |     |
| --- | ----- | --- | --- | ----- | --- | --- | --- | ----- | --- | --- | --- | ----- | --- |
Figure 5: CoDeR-Seq policy sensitivity measured by Figure6: CoDeR-Unionpolicysensitivitymeasuredby
Recall@10 on Negation and ExcluIR. Higher values Recall@10onNegationandExcluIR.Therelativefilter
indicatebetterpreservationofsatisfyingevidence. The percentile creates smoother high-recall regions, espe-
mapshowsthatrecallremainshighoverabroadlow- ciallywhenαkeepssufficienttopical-retrievalweight.
to-moderatethresholdregion,butcandropwhenthresh-
oldingbecomestoorestrictive.
|     |     |     |     |     |     |     | wrong    | constraint | direction. |     | The   | ablations | there-   |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | ---------- | --- | ----- | --------- | -------- |
|     |     |     |     |     |     |     | fore ask | whether    | CoDeR’s    |     | gains | come      | from the |
gioncanshiftwithscorecalibrationacrossdatasets.
proposeddecouplingoftopicalityandcompatibil-
CoDeR-Unionreplacesthisabsolutedecisionwith
|     |     |     |     |     |     |     | ity, or | from | simpler | alternatives |     | such as | using a |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---- | ------- | ------------ | --- | ------- | ------- |
relativefiltering,makingthepolicylessdependent
|     |     |     |     |     |     |     | stronger | topical | retriever, |     | replacing | the | compati- |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------- | ---------- | --- | --------- | --- | -------- |
onrawscorescaleandproducingsmootherbehav-
bilitysidewithanotherdenseencoder,relyingon
| ior across | Negation |     | and ExcluIR. | The | endpoint |     |     |     |     |     |     |     |     |
| ---------- | -------- | --- | ------------ | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
thetrainedconstraintencoderalone,orattachinga
behaviorofαisalsoconsistentwiththeintended
standardrerankerafterretrieval.
| mechanism: |     | largertopicalweightpreservesrecall, |     |     |     |     |        |       |           |     |         |         |       |
| ---------- | --- | ----------------------------------- | --- | --- | --- | --- | ------ | ----- | --------- | --- | ------- | ------- | ----- |
|            |     |                                     |     |     |     |     | Tables | 8 and | 9 examine |     | the two | encoder | roles |
whilestrongercompatibility-drivenfilteringdelays
|     |     |     |     |     |     |     | in this | decomposition. |     | In  | Group | 1, we | fix the |
| --- | --- | --- | --- | --- | --- | --- | ------- | -------------- | --- | --- | ----- | ----- | ------- |
earlyviolationexposure.
|     |     |     |     |     |     |     | compatibility-side |     |     | encoder | as CoDeR |     | and vary |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- | ------- | -------- | --- | -------- |
thetopicalencoderamongBGE-large,BGE-base,
G AblationStudy
|     |     |     |     |     |     |     | and BGE-small. |     |     | This setting | tests | whether | the |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | ------------ | ----- | ------- | --- |
Wefurtherconductablationexperimentstotestthe compatibility signal remains meaningful when
|            |       |           |           |        |      |      | the candidate |             | generator |         | changes.     | In  | Group 2, |
| ---------- | ----- | --------- | --------- | ------ | ---- | ---- | ------------- | ----------- | --------- | ------- | ------------ | --- | -------- |
| structural | claim | behind    | CoDeR     | rather | than | only |               |             |           |         |              |     |          |
|            |       |           |           |        |      |      | we fix        | the topical |           | encoder | as BGE-large |     | and re-  |
| to compare |       | component | strength. | The    | main | pa-  |               |             |           |         |              |     |          |
per argues that constraint-sensitive retrieval fails place the compatibility-side backbone with Con-
|     |     |     |     |     |     |     | triever | or miniLM. |     | This | setting | tests whether | a   |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---------- | --- | ---- | ------- | ------------- | --- |
whentopicalrelevanceandconstraintcompatibil-
|     |     |     |     |     |     |     | generic | semantic | encoder |     | can play | the same | role |
| --- | --- | --- | --- | --- | --- | --- | ------- | -------- | ------- | --- | -------- | -------- | ---- |
ityarecollapsedintoasinglesemantic-similarity
score: aviolatingdocumentcanbehighlytopical asaconstraint-compatibilityencodertrainedwith
because it mentions the same entities, attributes, satisfying–violatingcontrasts.
anddomainvocabulary,whilestillsupportingthe ThefirstinsightisthatEncoderAbehaveslikea
12

|     | Negation |     |     | ExcluIR |     | itysideisreplacedbyContrieverorminiLM,the |     |     |     |     |     |     |
| --- | -------- | --- | --- | ------- | --- | ----------------------------------------- | --- | --- | --- | --- | --- | --- |
1.0
|     |     |     |     |     |     | system | may still | rank | semantically |     | related | docu- |
| --- | --- | --- | --- | --- | --- | ------ | --------- | ---- | ------------ | --- | ------- | ----- |
0.55
| 0.8 |     |     |     |     |     | 0.50 ments,butitlosesthespecificpressurethatpushes |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- |
0.45 )retteb si rehgih( RVF
| 0.6 |     |     |     |     |     | 0.40 wrong-directionevidencedownward. |     |         |          |     | Thisexposes |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | ------- | -------- | --- | ----------- | --- |
| uat |     |     | uat |     |     |                                       |     |         |          |     |             |     |
|     |     |     |     |     |     | 0.35 the difference                   |     | between | semantic |     | relatedness | and |
0.4
0.30
|     |     |     |     |     |     | compatibility: |     | ContrieverandminiLMcanrecog- |     |     |     |     |
| --- | --- | --- | --- | --- | --- | -------------- | --- | ---------------------------- | --- | --- | --- | --- |
| 0.2 |     |     |     |     |     | 0.25           |     |                              |     |     |     |     |
0.20
|     |     |     |     |     |     | nizethatadocumentisaboutthesametopic, |     |     |     |     |     | yet |
| --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --- | --- |
0.0
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 theyarenottrainedtotreatsatisfyingandviolating
|        | alpha        |        |             | alpha |          |              |             |          |           |        |           |        |
| ------ | ------------ | ------ | ----------- | ----- | -------- | ------------ | ----------- | -------- | --------- | ------ | --------- | ------ |
|        |              |        |             |       |          | counterparts | as          | opposite | retrieval |        | outcomes. | The    |
| Figure | 7: CoDeR-Seq | policy | sensitivity |       | measured | by           |             |          |           |        |           |        |
|        |              |        |             |       |          | occasional   | improvement |          | of        | miniLM | on a      | narrow |
FVRonNegationandExcluIR.EachcellshowsFVR
settingshouldthereforenotbereadasevidencethat
| normalizedbyk |     | +1=11;valuescloserto1indicate |     |     |     |                                  |     |     |     |     |            |     |
| ------------- | --- | ----------------------------- | --- | --- | --- | -------------------------------- | --- | --- | --- | --- | ---------- | --- |
|               | max |                               |     |     |     | anydenseencodercansubstituteforE |     |     |     |     | ;rather,it |     |
C
thatthefirstviolationappearslaterintherankedlist,and
showsthataggressivereorderingcansometimesre-
| 1indicatesnoviolationwithinthetop10. |     |     |     |     | Theabsolute |     |     |     |     |     |     |     |
| ------------------------------------ | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
threshold τ can improve violation delay in effective moveviolationswhilealsoweakeningtheintended
regions, but its behavior depends on dataset-specific topical-preservationbehavior.
compatibility-scorecalibration. Together,Tables8and9supportthecentralde-
signchoiceofCoDeR.Themodularsplitisuseful
Negation ExcluIR because the two encoders answer different ques-
| 1.0 |     |     |     |     |     | tions: EncoderAaskswhetheradocumentisabout |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | --- |
0.55
| 0.8 |     |     |     |     |     | 0.50 thequery,whileEncoderBaskswhetherthedoc- |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --------------------------------------------- | --- | --- | --- | --- | --- | --- |
0.45 )retteb si rehgih( RVF
elitnecrep retlif elitnecrep retlif ument is compatible with the query’s constraint
| 0.6 |     |     |     |     |     | 0.40 |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
0.35
|     |     |     |     |     |     | direction. | The | observed | robustness |     | across | topi- |
| --- | --- | --- | --- | --- | --- | ---------- | --- | -------- | ---------- | --- | ------ | ----- |
0.30
| 0.4 |     |     |     |     |     | cal backbones |     | and the | degradation |     | under | generic |
| --- | --- | --- | --- | --- | --- | ------------- | --- | ------- | ----------- | --- | ----- | ------- |
0.25
| 0.2 |     |     |     |     |     | 0.20 compatibility |     | replacements |     | jointly | indicate | that |
| --- | --- | --- | --- | --- | --- | ------------------ | --- | ------------ | --- | ------- | -------- | ---- |
0.1
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 constraint-awareretrievalisnotobtainedbymerely
|     | alpha |     |     | alpha |     | scalingtopicalretrieval. |     |     | Itrequiresalearnedcom- |     |     |     |
| --- | ----- | --- | --- | ----- | --- | ------------------------ | --- | --- | ---------------------- | --- | --- | --- |
Figure8: CoDeR-Unionpolicysensitivitymeasuredby patibilitydimensionthatiscomposedwithtopical
FVRonNegationandExcluIR.EachcellshowsFVR
coverageatretrievaltime.
| normalizedbyk | max | +1=11;valuescloserto1indicate |     |     |     |     |     |     |     |     |     |     |
| ------------- | --- | ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thatthefirstviolationappearslaterintherankedlist,and G.1 CompatibilityEncoderScoreSeparation
| 1indicatesnoviolationwithinthetop10. |     |     |     |     | Therelative |     |     |     |     |     |     |     |
| ------------------------------------ | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
Wefurtherinspectwhetherthetrainedcompatibil-
filterpercentileproducesgradualchangesinviolation
delay: strongerfilteringtendstopushviolationslater, ity encoder changes the score geometry relative
|     |     |     |     |     |     | to the base | BGE | encoder. | Figure |     | 9 compares | the |
| --- | --- | --- | --- | --- | --- | ----------- | --- | -------- | ------ | --- | ---------- | --- |
whileweakerfilteringpreservesalargercandidateset.
|     |     |     |     |     |     | trained    | BGE constraint |             | encoder | with              | BGE-large |     |
| --- | --- | --- | --- | --- | --- | ---------- | -------------- | ----------- | ------- | ----------------- | --------- | --- |
|     |     |     |     |     |     | on ExcluIR | using          | query-level |         | satisfying-minus- |           |     |
coveragecomponent,notthesourceofconstraint violatingmarginsandrawcompatibility-scoredis-
direction. Strongertopicalencodersgenerallypre- tributions. The trained encoder shifts the mar-
serve ordinary BEIR-style retrieval better on Sci- gindistributiontowardpositivevaluesandassigns
| Fact and | NFCorpus, | which |     | is expected | because |     |     |     |     |     |     |     |
| -------- | --------- | ----- | --- | ----------- | ------- | --- | --- | --- | --- | --- | --- | --- |
highercompatibilityscorestosatisfyingevidence
they determine which topically plausible candi- than to violating evidence. By contrast, the base
datesenterthepool. However,whenEncoderBis BGE encoder shows more overlap between the
keptasthetrainedcompatibilityencoder,changing
|     |     |     |     |     |     | two evidence |     | types, | consistent | with | its role | as a |
| --- | --- | --- | --- | --- | --- | ------------ | --- | ------ | ---------- | ---- | -------- | ---- |
EncoderAproducesmuchsmallershiftsinearly- topicalsemanticencoderratherthanaconstraint-
violationbehaviorthanchangingEncoderB.This compatibility scorer. This analysis complements
isimportantbecauseitrulesouttheinterpretation
thecomponentablationsbelowbyshowingthatthe
| that CoDeR | works | simply | because | BGE-large |     | is  |     |     |     |     |     |     |
| ---------- | ----- | ------ | ------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
trainedEncoderBcontributesaseparablecompati-
a strong retriever. A strong topical encoder can bilitydirectioninthescoringspace.
supplybettercandidatecoverage,butitdoesnotby Table 10 isolates the role of the trained com-
itselfdecidewhetheratopicallysimilardocument
|     |     |     |     |     |     | patibility | encoder | on  | NevIR, | where | the | relevant |
| --- | --- | --- | --- | --- | --- | ---------- | ------- | --- | ------ | ----- | --- | -------- |
satisfiesorviolatestheuser’sconstraint. failureisnottopicmismatchbutopposite-direction
ThesecondinsightisthatEncoderBisnotjust negationevidenceappearingtooearly. Thisabla-
another semantic reranker. When the compatibil- tionseparatesthreepossibilitiesthatareconflated
13

EncoderA EncoderB SciFact NFCorpus Antonym Negation Exclusion
nDCG↑ MAP↑ nDCG↑ MAP↑ V@2↓ V@3↓ V@5↓ V@10↓ FVR↑ V@2↓ V@3↓ V@5↓ V@10↓ FVR↑ V@2↓ V@3↓ V@5↓ V@10↓ FVR↑
Group1:FixEncoderB=CoDeR-Seq,varyEncoderA
BGE-large CoDeR-Seq 74.77 69.93 37.24 28.00 59.80 76.47 91.18 99.02 2.95 50.98 73.53 84.31 92.16 3.53 50.00 60.58 74.04 84.62 4.26
BGE-base CoDeR-Seq 73.88 68.88 36.68 27.54 60.78 74.51 93.14 99.02 2.94 51.96 74.51 84.31 94.12 3.42 50.98 64.71 77.45 86.27 4.05
BGE-small CoDeR-Seq 69.68 64.92 34.82 25.57 58.82 77.45 90.20 99.02 2.99 51.96 74.51 84.31 92.16 3.50 49.02 63.73 75.49 86.27 4.14
Group2:FixEncoderA=BGE-large,varyEncoderB
BGE-large Contriever 74.77 69.93 37.24 28.00 80.39 94.12 99.02 100.00 2.26 71.57 83.33 95.10 98.04 2.66 66.67 82.35 92.16 97.06 2.77
BGE-large miniLM 74.77 69.93 37.24 28.00 76.47 86.27 92.16 99.02 2.64 59.80 73.53 84.31 89.22 3.51 45.20 56.86 79.41 88.24 4.09
Table 8: Ablation study for CoDeR-Seq. nDCG and MAP are reported at @10. Group 1 fixes Encoder B as
CoDeR-SeqandvariesEncoderAamongBGE-large,BGE-base,andBGE-small. Group2fixesEncoderAas
BGE-largeandvariesEncoderBamongContrieverandminiLM.TheBGE-large/CoDeR-Seqrowattheendof
Group1servesasthesharedbaselineforbothgroups.
EncoderA EncoderB SciFact NFCorpus Antonym Negation Exclusion
nDCG↑ MAP↑ nDCG↑ MAP↑ V@2↓ V@3↓ V@5↓ V@10↓ FVR↑ V@2↓ V@3↓ V@5↓ V@10↓ FVR↑ V@2↓ V@3↓ V@5↓ V@10↓ FVR↑
Group1:FixEncoderB=CoDeR-Union,varyEncoderA
BGE-large CoDeR-Union 74.91 70.17 36.36 27.12 52.94 76.47 93.13 98.04 3.00 49.02 72.55 89.22 90.20 3.47 48.08 61.54 75.96 81.73 4.37
BGE-base CoDeR-Union 74.05 69.06 36.59 27.40 53.92 75.49 96.08 98.04 2.93 46.08 72.55 90.20 90.20 3.50 49.02 65.69 79.41 84.31 4.15
BGE-small CoDeR-Union 69.92 65.28 35.31 25.86 52.94 76.47 94.12 98.04 2.98 47.06 72.55 89.22 90.20 3.49 49.02 62.75 79.41 84.31 4.17
Group2:FixEncoderA=BGE-large,varyEncoderB
BGE-large Contriever 68.27 64.36 32.86 23.34 74.51 90.20 99.02 100.00 2.19 46.47 80.39 85.10 89.22 3.00 55.88 79.41 90.20 94.12 3.10
BGE-large miniLM 71.38 66.96 33.44 23.64 66.67 80.39 94.12 97.06 2.81 51.96 69.61 87.25 88.24 3.63 33.33 47.06 70.59 77.45 4.99
Table9: AblationstudyforCoDeR-Union. nDCGandMAParereportedat@10. Group1fixesEncoderBas
CoDeR-UnionandvariesEncoderAamongBGE-large,BGE-base,andBGE-small. Group2fixesEncoderAas
BGE-largeandvariesEncoderBamongContrieverandminiLM.TheBGE-large/CoDeR-Unionrowattheendof
Group1servesasthesharedbaselineforbothgroups.
|             |             | 6         |                                        |         |                             |     |           | NevIR |           |
| ----------- | ----------- | --------- | -------------------------------------- | ------- | --------------------------- | --- | --------- | ----- | --------- |
| 3.0         |             |           | Satisfying evidence Violating evidence |         | Model                       |     |           |       |           |
| 2.5         |             | 5         |                                        |         |                             |     | Recall@5↑ | V@1↓  | FVR↑      |
| ytisneD 2.0 |             | ytisneD 4 |                                        |         | BGE-large                   |     | 85.85     |       | 19.6 3.70 |
| 1.5         |             | 3         |                                        |         | TrainedBGEConstraintEncoder |     |           | 79.2  | 14.2 4.45 |
| 1.0         |             | 2         |                                        |         | CoDeR-Seq                   |     | 81.65     |       | 15.4 4.26 |
|             |             |           |                                        |         | CoDeR-Union                 |     |           | 81.7  | 13.9 4.28 |
| 0.5         |             | 1         |                                        |         |                             |     |           |       |           |
| 0.0         |             | 0         |                                        |         |                             |     |           |       |           |
| -0.4 -0.2   | 0.0 0.2 0.4 | 0.6 0.2   | 0.4                                    | 0.6 0.8 |                             |     |           |       |           |
Margin Compatibility score Table10: ComparisonofBGE-large,thetrainedBGE
4.0
|     |     |     | S a t i s f y in g  e v id e n ce |     | constraintencoder,CoDeR-Seq,andCoDeR-Unionon |     |     |     |     |
| --- | --- | --- | --------------------------------- | --- | -------------------------------------------- | --- | --- | --- | --- |
| 3.5 |     | 5   | Vi o l a t i n g  e vi d e n ce   |     |                                              |     |     |     |     |
3.0
|     |     | 4   |     |     | NevIR. |     |     |     |     |
| --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- |
2.5
| ytisneD |     | ytisneD 3 |     |     |     |     |     |     |     |
| ------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
2.0
1.5
2
1.0
|     |     | 1   |     |     | shows that | polarity supervision |     | can | reshape the |
| --- | --- | --- | --- | --- | ---------- | -------------------- | --- | --- | ----------- |
0.5
| 0.0            |             | 0       |                     |             | scorespacetowardcompatibility,butusingitalone |     |     |     |     |
| -------------- | ----------- | ------- | ------------------- | ----------- | --------------------------------------------- | --- | --- | --- | --- |
| -0.3 -0.2 -0.1 | 0.0 0.1 0.2 | 0.3 0.4 | 0.5 0.6             | 0.7 0.8 0.9 |                                               |     |     |     |     |
|                | Margin      |         | Compatibility score |             |                                               |     |     |     |     |
removespartofthetopical-retrievalfunctionthata
Figure9: Compatibility-scoreseparationonExcluIR.
generalretrieverstillneeds.
ThetoprowusesthetrainedBGEconstraintencoder, TheCoDeRrowsarethereforethecriticalcom-
andthebottomrowusesthebaseBGE-largeencoder.
parisonratherthanasimplemiddlepointbetween
Leftpanelsshowquery-levelsatisfying-minus-violating
|     |     |     |     |     | BGE-large | and the constraint |     | encoder. | CoDeR- |
| --- | --- | --- | --- | --- | --------- | ------------------ | --- | -------- | ------ |
margins;rightpanelsshowrawcompatibility-scoredis-
tributions for satisfying and violating evidence. The SeqandCoDeR-Unionusethecompatibilitysignal
asaretrieval-sidecontroloveratopicalcandidate
trainedcompatibilityencoderproducesmorepositive
query-levelmarginsandclearerscoreseparationthan process, sothelearnedconstraintdirectionisnot
thebasetopicalencoder.
|     |     |     |     |     | treated                              | as a replacement | for | topicality | but as an |
| --- | --- | --- | --- | --- | ------------------------------------ | ---------------- | --- | ---------- | --------- |
|     |     |     |     |     | additional                           | axis for ranking | and | filtering. | This ex-  |
|     |     |     |     |     | plainstheintendedtrade-offinTable10: |                  |     |            | thefull   |
inthefullsystem: astrongtopicalencoderalone, systems give up some pure topical recall relative
thetrainedconstraintencoderalone,andtheinte- to BGE-large, but they reduce early violation ex-
gratedCoDeRpolicies. BGE-largerepresentsthe posurewhileretainingsubstantiallymoreretrieval
first case: it preserves topical recall because it is coveragethanacompatibility-onlyinterpretation
optimized for semantic matching, but this objec- would require. The insight is that E C is useful
tive does not directly encode which member of a because it becomes part of a two-signal retrieval
negationpairsatisfiesthequery. ThetrainedBGE policy,notbecauseastandaloneconstraintencoder
constraint encoder represents the second case: it shouldreplacethetopicalretriever.
14

|     |     |     |     | Antonym |     | Negation |     | Exclusion |     | NevIR |     | ExcluIR |
| --- | --- | --- | --- | ------- | --- | -------- | --- | --------- | --- | ----- | --- | ------- |
Model
nDCG@5↑ FVR↑ nDCG@5↑ FVR↑ nDCG@5↑ FVR↑ nDCG@5↑ FVR↑ nDCG@5↑ FVR↑
BGE-large+BGEReranker 80.00 4.32 79.57 2.90 75.34 2.91 60.48 4.00 83.87 1.79
CoDeR-Seq+BGEReranker 80.43 4.42 78.83 3.42 75.07 3.63 60.78 4.28 85.35 4.11
CoDeR-Union+BGEReranker 80.25 4.45 78.89 3.57 75.16 3.95 60.19 4.28 86.06 4.84
Table11: RerankingcomparisonacrossAntonym,Negation,Exclusion,NevIR,andExcluIRusingnDCG@5and
FVR.
Table11testsadifferentalternativeexplanation: Method AnswerAccuracyRate↑ FVRAvg.↑
perhapsastandardrerankercanrepairconstraintvi-
|     |     |     |     |     |     |     |     | BM25 |     | 15.00% |     | 2.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | ------ | --- | ---- |
olationsafterretrieval,makingcompatibility-aware BGE 30.00% 3.45
|           |              |     |          |          |       |            |     | Contriever |     | 25.00% |     | 3.15 |
| --------- | ------------ | --- | -------- | -------- | ----- | ---------- | --- | ---------- | --- | ------ | --- | ---- |
| retrieval | unnecessary. |     | This     | ablation |       | is aligned |     |            |     |        |     |      |
|           |              |     |          |          |       |            |     | HyDE       |     | 30.00% |     | 3.00 |
| with the  | methodology  |     | in which |          | CoDeR | operates   |     |            |     |        |     |      |
|           |              |     |          |          |       |            |     | NS-IR      |     | 25.00% |     | 3.45 |
before optional downstream rerankers or genera- DEO 30.00% 3.35
tors. ThequestionisthereforenotwhetheraBGE CoDeR-Seq 40.00% 3.80
|     |     |     |     |     |     |     |     | CoDeR-Union |     | 35.00% |     | 3.80 |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------ | --- | ---- |
rerankercanimprovesemanticrankingquality,but
whetheratopicalrerankercanundotheexposure
|                         |     |     |          |     |          |         | Table12: |     | End-to-endRAGstressteston20randomly |     |     |     |
| ----------------------- | --- | --- | -------- | --- | -------- | ------- | -------- | --- | ----------------------------------- | --- | --- | --- |
| of constraint-violating |     |     | evidence |     | that has | already |          |     |                                     |     |     |     |
sampledNevIRqueries.
enteredthecandidatelist.
| The | comparison |     | suggests | that | reranking | and |     |     |     |     |     |     |
| --- | ---------- | --- | -------- | ---- | --------- | --- | --- | --- | --- | --- | --- | --- |
compatibility-aware retrieval address different pled NevIR question-style queries. We evaluate
|          |               |     |           |     |      |       | whether |     | the top retrieved | evidence | supports | the |
| -------- | ------------- | --- | --------- | --- | ---- | ----- | ------- | --- | ----------------- | -------- | -------- | --- |
| parts of | the pipeline. |     | BGE-large |     | plus | a BGE |         |     |                   |          |          |     |
satisfyinganswerdirection,whilekeepingthecor-
| reranker | remains | strong | in  | nDCG | because | both |     |     |     |     |     |     |
| -------- | ------- | ------ | --- | ---- | ------- | ---- | --- | --- | --- | --- | --- | --- |
components reward semantic usefulness and top- pus,queries,retrievaldepth,generator,andprompt
fixedacrossmethods.
| ical match. | However, |     | this | does | not guarantee | a   |     |     |     |     |     |     |
| ----------- | -------- | --- | ---- | ---- | ------------- | --- | --- | --- | --- | --- | --- | --- |
Table12isasmalldownstreamproberatherthan
laterfirstviolation,especiallyonsettingssuchas
|     |     |     |     |     |     |     | afullRAGbenchmark. |     |     | ItconnectsFVRtogenera- |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- | ---------------------- | --- | --- |
Negation,Exclusion,andExcluIRwheretheviolat-
ingevidenceisintentionallytopicallyclosetothe tionrisk: earlyviolationscangroundthegenerator
inevidencethatistopicallyrelevantbutdirection-
| satisfyingevidence. |       |       | Incontrast,applyingthesame |      |             |      |            |     |                                  |     |     |     |
| ------------------- | ----- | ----- | -------------------------- | ---- | ----------- | ---- | ---------- | --- | -------------------------------- | --- | --- | --- |
|                     |       |       |                            |      |             |      | allywrong. |     | CoDeRdelaysthiscontextcontamina- |     |     |     |
| reranker            | after | CoDeR | starts                     | from | a candidate | list |            |     |                                  |     |     |     |
whoseriskstructurehasalreadybeenchangedby tionandmoreoftenplacessatisfyingevidencebe-
|                       |     |     |                        |     |     |     | foreconflictingevidence. |     |     | Thesepreliminaryprobe |     |     |
| --------------------- | --- | --- | ---------------------- | --- | --- | --- | ------------------------ | --- | --- | --------------------- | --- | --- |
| compatibilityscoring. |     |     | Thererankercanthenpre- |     |     |     |                          |     |     |                       |     |     |
resultsareconsistentwiththeretrieval-sidetrend.
serveorrefinerelevancewithouthavingtodiscover
theconstraintdirectionfromscratch.
H.1 DownstreamProbePrompt
ThisresultclarifiestheroleofCoDeRinaRAG-
stylepipeline. CoDeRisnotproposedasareplace- You are a question answering assistant.
mentforalldownstreamreranking;itisaretrieval- You are given retrieved passages that are
|     |     |     |     |     |     |     |     | relevant | to the question. |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------------- | --- | --- | --- |
sidemechanismforreducingharmfulearlyexpo-
surebeforelatercomponentsconsumethecontext. Answer the question based on the retrieved
passage.
Theablationthereforesupportsthebroaderclaim
Question:
| ofthepaper: | constraintcompatibilityisnotaby- |     |     |     |     |     |     |     |     |     |     |     |
| ----------- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{query}
| product    | of stronger |      | topical       | retrieval | or    | standard |     |           |          |     |     |     |
| ---------- | ----------- | ---- | ------------- | --------- | ----- | -------- | --- | --------- | -------- | --- | --- | --- |
| reranking. | It          | must | be introduced |           | as an | explicit |     | Retrieved | passage: |     |     |     |
{retrieved_doc}
| retrieval | signal | that | separates | topically |     | plausible |     |     |     |     |     |     |
| --------- | ------ | ---- | --------- | --------- | --- | --------- | --- | --- | --- | --- | --- | --- |
Rules:
satisfyingevidencefromtopicallyplausibleviola-
|     |     |     |     |     |     |     |     | - Output | only the | short answer. |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | ------------- | --- | --- |
tions.
|     |     |     |     |     |     |     |     | - Do not | output any | explanation, | punctuation, |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | ------------ | ------------ | --- |
|     |     |     |     |     |     |     |     | or extra | text.      |              |              |     |
H PreliminaryRAG-OrientedProbe
| To connect   | retrieval-side |     |     | violation | control       | with |     |     |     |     |     |     |
| ------------ | -------------- | --- | --- | --------- | ------------- | ---- | --- | --- | --- | --- | --- | --- |
| answer-level | behavior,      |     | we  | conduct   | a lightweight |      |     |     |     |     |     |     |
End-to-EndRAGstressteston20randomlysam-
15
