
Cet enregistrement fait office de rodmap sur mon travail d'optimisation de modèle. Dans le cadre du projet de dsai. Où on était censé faire du transpid optimization? C'est-à-dire l'objectif était de prendre des modèles. De détection. De compétition pour qui font la détection d'objets? Avec. Le jeu de données coco et essayer justement d'optimiser ces modèles pour qu'ils puissent faire de l'influence beaucoup plus rapidement que ce qu'ils nous font.

J'étais en charge de plusieurs modèles notamment les modèles Retina. Avec le backbone, r50 et r101. La première difficulté avec cette famille de modèle là, c'est que Retina r101. N'était pas disponible. Retina r50 était disponible sur torche vision la version 2 des poids avec les poids tout était disponible.

Les poids que j'ai trouvé pour ce modèle était lié à la bibliothèque de ta grands faite par détecteur de faite par. Facebook Facebook ou search? J'ai eu un travail avec cette bibliothèque là sauf que très souvent elle était. Elle était assez instable pour les versions vu qu'elle n'est plus maintenue depuis déjà.

Quelques années depuis 2008, 3 ou 4 ans déjà. Donc les autres modèles sur lesquels je travaillais c'était notamment F COS, fco SS R50. Avec le gbagboard 50 mais aussi. Efficient d'être det. Notamment D4 D5 D6. Donc? Mon travail était assez simple. La première phase du travail c'était de faire le profané pour le profil.

Les outils soumis à notre appréciation était principalement torche profiler il permettait justement de. Faire de l'inférence des modèles. Et être à même de savoir. À peu près, quelles opérations prennent combien de temps? J'ai fait l'exploration de dancehall pour ça. J'ai utilisé des vidéos YouTube produit par. À la chaîne de paille touch où il parlait justement de comment est-ce qu'on utilise et à quoi ça sert?

J'ai suivi des tutoriels. YouTube dessus produite par la chaîne et j'ai justement pris l'amour avec outil. J'ai écrit ma premier Street. Grâce à? Et j'ai fait mon premier profiling. Se faisait justement comme je l'ai dit avec le note à cette coco date à celle de validation coco Donc? Une fois le profil fait, j'ai analysé les traces sur tenseur Word et j'étais à Sainte admiratif de voir à quel point on pouvait suivre allant des traces aux opérations aux opérations alimentaires.

Voir à peu près. Comment est-ce que les données transident dans les différentes couches tant que ça met dans chaque couche? Comment est-ce que les couches du modèle sont hiérarchisées sont regroupés? C'était assez passionnant de constater ça. Très rapidement, je me suis rendu compte que les modèles n'avaient pas tous la même taille.

Que par exemple? Et aussi que ce n'était pas forcément la taille du modèle qui. Définissait quelle était la vitesse? Quelle était la performance du modèle? On pouvait tomber sur des modèles beaucoup plus petits qui avaient. Qui était un peu meilleur qui avait de bien meilleures performances que d'autres modèles.

Donc après cette phase d'exploration j'ai essayé de démarrer avec le travail qui avait été demandé. Ça diffère le profiling dans un cadre bien défini donc. J'ai écrit un script qui permettait justement de faire du conseil ligne. Et avec l'aide d'un modèle dia notamment j'ai mis ici. J'ai fait mes premières expérimentations sur cola sur le descart de cola ou justement je.

J'étais capable d'extraire les temps individuels sur chaque. Couche sur chaque couche de modèle de façon vraiment indépendante. C'était assez difficile. Les principales difficultés que j'ai rencontrées ici. Était le fait qu'il y avait une sorte de manque que le standard. Donc comme j'ai dit au départ retrouver les modèles n'étaient pas forcément évidents à part les autres modèles.

Les Retina et foncé assez difficile parce que il n'y a pas vraiment d'endroits où tout est bien centralisé. Il n'y a pas forcément de standards les chercheurs quand ils finissent d'écrire les papiers lorsqu'il y a des ports de modèles. Il n'y a pas ces modèles là ne sont par exemple pas hébergé sur un groupe face comme les modèles comme les llm de nos jours.

Donc trouver à peu près des sources fiables étaient assez difficiles. Donc j'ai fait du profiling avec les scripts ça c'était comme recommandé par le prof où on essaie de savoir à peu près. Quel est le temps mis dans différentes couches? Et à partir de ça? J'ai pu avoir une vue globale de comment est-ce que mon système fonctionnait?

La deuxième phase consistait à faire les optimisations donc non seulement d'abord analyser analyser complètement les traces les résultats du coup, Fanny pour savoir quels étaient les leviers qu'on pourra qu'on pouvait emprunter pour améliorer. Quels étaient les faiblesses qu'on pouvait rencontrer? Quels éléments qu'on pouvait détecter pour faire des améliorations?

Donc en lisant les traces, je me suis rendu compte par exemple que Les batch non? Les couches qui faisaient de la bâche noire était. Très souvent collé avec des couches qui faisaient. De la convolution or, étant donné que nous faisons de référence et que nous mettons dans une situation où nous sommes en production, le badge était de 1 pour nos travaux et lorsque le Barça est de 1 la normalisation des données.

Perd son sens parce que. Ça devient juste une très simple transformation linéaire pour les coefficients sont constants donc je me suis rendu compte qu'on pouvait. Si j'ai cela et ne plus avoir besoin de passer par la couche bash normalisation ne peut avoir besoin de l'exécuter comme elle se fait et fusionner cela avec.

Directement avec les couches de coincolutions. On sait également rendu compte avec l'appui de notre enseignant de notre encadreur. Que certaines fonctions d'activation prenaient beaucoup plus de temps que d'autres, notamment les fonctions d'activation comme cellule. Et à ce niveau là? Il fallait trouver le moyen, il fallait déjà comprendre pourquoi est-ce que c'est lui prenait beaucoup plus de temps que rélu?

Comprendre comment est-ce qu'on pouvait faire pour y remédier? Si lui prenait beaucoup plus de temps que j'ai lu notamment parce que c'est lui faisait un calcul de. D'exponentielle ce qui est plus nécessaire c'est qui? Est aussi le cas de la sigmoïde mais en plus de faire une exponentielle il y avait les transferts mémoire qui était.

Beaucoup plus sollicité parce qu'on faisait le produit de l'entrée par l'exponentielle de l'entrée de par la sigmoïde de l'entrée. Pardon et lorsqu'on fait le produit de l'entrée par la sécurité de l'entrée il y a deux, il y a d'abord la qu'il faut calculer. C'est à l'étranger de mémoire pour ça, puis également l'entrée en elle-même.

Et une solution justement. C'était de regarder la fonction d'activation. Regardez la couche. Après laquelle elle est placée et voir dans quelle mesure fusionner ces deux couches là pour ne faire qu'un seul noyau. Également, dans la phase d'exploration je me suis rendu compte. Je me suis rendu compte que. Que aucun!

Aucune des opérations n'utilisait le cœur CUDA. Utiliser le canal CUDA. C'était assez intrigant pour moi parce que je me disais la raison pour laquelle des GPU. C'est justement pour pouvoir utiliser ces cannelle là. Donc si aucune opération on utilise l'Internet CUDA ça veut dire qu'on utilise juste la réalisation juste les corps et pas encore l'Éternel ne pas utiliser la guerre, ça veut dire qu'il y a un levier en échangeant avec l'encadreur sur ce point, il m'a fait comprendre que.

C'était quelque chose qui allait venir plus tard. Et plus tard justement. J'ai essayé de regarder. Plus tard justement je m'en suis rendu compte bien après lorsque Du fait des difficultés que je rencontrais avec eux colap parce que à certains moments ça s'arrêtait Google collapse s'arrêtait, on pouvait penser de les expériences qui durent beaucoup trop longtemps.

Les espérance de plus de 3 à 4h par là ou à ce moment, ça planté tout simplement où il arrivait que un moment, la RAM plante ou un moment le. L'instant ce s'arrête et on doit redémarrer tout le travail qu'on avait déjà qu'on avait déjà fait. Donc je me suis dit pourquoi ne pas travailler sur mon laptop personnel vu que sur mon laptop j'ai un GPU.

Pourquoi ne pas travailler sur mon laptop faire les petites expériences à petite échelle à très petite échelle sur mon laptop. Et ce n'est que quand le code est totalement fait bien implémenter que je devrais le déployer justement et l'expérimenter sur une plateforme comme qu'on a et dans cette logique là j'ai testé.

J'ai essayé de travailler également sur mon laptop. Et en essayant de travailler sur mon laptop quand j'ai fait du profil ligne déjà première élément, je me suis rendu compte que ça allait beaucoup plus vite que sur. Puis je quand je me suis renseigné, je me suis rendu compte que c'est parce que le GPU qui était dans mon laptop.

Était d'une relation beaucoup plus récente que ce qui arrive sur qu'on a et quand je suis allé sur dancer word pour regarder les traces. De certaines sorboard pour regarder les traces. De profilage je me suis rendu compte que. Dans le cas de mon laptop naturellement. Il y avait certains canal CUDA qui était utilisé.

Naturellement! Quand j'en ai parlé justement à l'enseignant, il m'a fait comprendre que c'est à cause des générations pour certaines générations, il y a certaines facilités qui sont implémentées donc on peut ici espérer que si on passe d'un. Si naturellement de façon générale comme. Si on si les travaux que nous avons fait sur un T4 sur un T4 si nous les faisons sur, j'ai pu beaucoup plus récent de pouvoir avoir de bien meilleurs résultats, non pas parce que.

Non pas forcément parce que il y a plus de mémoire ou plus de corps CUDA mais aussi et surtout parce que les technologies sont beaucoup plus récentes et sont beaucoup plus adaptés pour porter ce genre de problème. Donc?

Pendant ma phase de d'analyse pour trouver les leviers sur lesquels je pouvais, je pouvais justement améliorer le travail. En discutant justement en faisant la recherche documentaire, je me suis rendu également compte que. C'était possible. Il y avait torche pasitoch à ce qu'on appelle un graphe de calcul et compliqué.

Je suis grave et ce graphe de calcul là. C'était un moyen pour lui justement de faire les calculs et si on réussissait justement à afficher ce gratte de calcul là et à trouver des raccourcis, ça pourrait permettre d'optimiser non seulement les allers-retours au niveau de la mémoire mais aussi.

Gagnant en temps en terme d'opération à réaliser. Donc pour faire ça je me suis rendu compte qu'il y avait les outils comme Cuba graphe. Dans script qui justement ne venait pas forcément, ne fonctionnait pas forcément de la fonction n'est pas forcément la même façon. Grâce à d'autres expériences je savais déjà, je comptais déjà l'existence de torche campagne.

Et aussi grâce. Orientations de notre encadreur. J'ai donc essayé de mener quelques petites expériences lorsque nous sommes passés à la fois suivante. Donc dès la phase 3 lorsqu'il fallait faire des appliquer, les optimisations, les principales optimisation que j'ai testé c'était que la grâce. Et torche compagne parce qu'au départ j'avais du mal à utiliser ton script.

J'étais encore sur cola. J'étais encore sur je faisais des allers-retours entre cola. Google collab. Un autre problème qui s'est posé c'est que depuis la phase lorsqu'on faisait encore le. Lorsqu'on faisait encore le benchmarking des modèles. J'avais du mal à évaluer dans ma. C'est vrai que j'utilisais du code généré par hier, mais la façon d'évaluer la marque causer problème lorsqu'on passait d'un modèle, une famille de modèle à l'autre.

C'était ceci était principalement dû au fait que la façon de nommer les classes variées d'une classe d'un ensemble de modèle à l'autre. Le jeu de données coco Coco que nous utilisions avec? Et les 80 classes c'était pas nommés de la même façon dans tout pour tous les. Pour tous les modèles, certains modèles nous met de 0 à 79 d'autres nommé de 1 à 80 d'autres domaines de 1 à 90 et en plus de ça et avait certains numéros qui n'existaient pas notamment le 11, le 12 13 et certaines plages de valeurs justement qui n'était pas utilisé.

Et ce décalage la faisait justement qu'il y avait des erreurs également dans la façon de calculer la précision, les calculs de précision, les outils en question ne venaient pas. Les bibliothèques ne venait pas avec des outils assez assez précis pour calculer ses précisions là. Donc? Une fois que nous sommes passés en phase 3, j'ai décidé de réécrire à peu près tout le code à la main par moi-même sans utiliser d'outilia et avant de le faire, je me suis mis à faire de la recherche.

Tout simplement, j'essaie de chercher par moi-même où est-ce que je pouvais trouver des poids sur mes modèles? C'est là où j'ai découvert moi-même. Quels outils? Comment est-ce que le jus de donner coco va fonctionner? Comment est-ce que d'autres personnes les utilisent souvent et je suis tombé sur?

Via les électrons grâce à des technologies découvert qu'il y avait, coco va laver le jeu de données coco avec de quoi évaluer. Justement la map la précision. Donc grâce aux dépenses que j'ai trouvé, j'ai. J'ai pu écrire un nouveau compte qui permettait justement non seulement il était beaucoup mieux structuré mais permettait de faire de façon beaucoup plus simple, l'évaluation le benchmarking le profil de tous les modèles de façon vraiment industrielle.

Donc j'ai un ré architecturé tout mon travail. Et en me basant sur ces références là. Je me suis dit au lieu d'essayer de d'avoir un code. D'avoir un cours de très abstrait qui gère tous les cas de figures en même temps autant des composés au maximum et faire du code spécifique pour chaque modèle.

Donc j'ai écrit. Je compte spécifique pour chaque modèle pour régler chacun des problèmes. Que ce soit du profiling que ce soit du benchmarking que ce soit de l'évaluation de précision. Pour cette fois-ci donc. Commencer à expérimenter à faire des expériences une par une donc. D'abord sur mon laptop mais certaines certains outils d'optimisation du marché pas sur mon laptop.

Ensuite chocolat. Donc? J'ai eu pas mal de difficultés avec une soirée qui était assez capricieux. En me renseignant sur le modèle je me suis rendu compte que. Le modèle qui était le plus propice à l'optimisation chez moi, c'était fishantenet parce que Parce que comme le nom l'indique, ce sont des modèles qui sont conçus justement pour pouvoir être optimisés au maximum.

Comment est-ce qu'il fonctionne? Comment est-ce que déjà de leur architecture ne contient que des? Couche linéaire. Des couches linéaires? Et des couches minières et convolution et des fonctions d'activation principalement c'est avec des normalisations. Par contre mes deux autres modèles Fos et Retina. Contenait des opérations particulières qui sont pouvait pas être portées sur GPU.

Notamment les MMS c'est-à-dire. Notamment les MMS. C'était c'est une sorte d'algorithme qui permet justement de trouver la plus petite boîte qui englobe. De trouver une boîte qui en cas de trouver de déterminer la boîte qui encadrera qu'en sélectionnera comme étant la boîte encadrant le

Quand sélectionnera comme étant la boîte encadrant l'objet détecté. Ou alors les BFM qui étaient contenues le fpn qui était le dfpn qui était contenu dans F course. Quant à lui contenait des opérations qui n'étaient pas forcément. Numérisable. C'est à dire des opérations? Un peu croisé c'était notamment l'une des.

Des innovations qui venaient avec eux le modèle après en lisant l'article fondateur de FC. Pour ce qui est de rétine, âgée également appris beaucoup sur son affection. Lisons le En lisant.

En lisant l'article fondateur donc après les premières optimisations. J'ai eu pas mal d'erreurs. Parfois, dans les erreurs, il y avait des suggestions sur comment est-ce que je pouvais résoudre ces erreurs? Parfois c'était juste des incompatibilités et autres et donc je me suis rendu compte justement qu'il y avait des valeurs qui pouvaient être figées dans les données.

Ça c'était des recommandations en fait. Une fois de plus de retour grâce officiel. Modèle de langage j'ai pu écrire du code beaucoup plus rapidement. Itérer beaucoup plus rapidement pour résoudre ces problèmes là à la chaîne. Étant donné que travailler sucollab devenait pénible. Notamment parce que lorsqu'on veut lancer du code pour aller dormir.

C'est difficile. J'ai donc décidé de faire la recherche d'autres outils qui pouvait me permettre d'avoir du GPU gratuitement l'été. 5. Je suis tombé sur une plateforme modèle.com qui me permettait justement de le faire. Donc? Bonne nouvelle! Façon de fonctionner était assez simple j'écrivais du code. Et je l'exécutais à distance grâce à un modèle et habitants, il faisait pour moi d'écrire le code sur mon laptop.

Ensuite écrit un script. Qui va permettre d'exécuter le code et la fonction va s'exécuter directement en direct. Modèle va lancer une machine virtuelle va initialiser avec les capacités que j'ai données et j'ai donné exactement les mêmes capacités qu'on avait sur un. Un T4? De cola. Et ça permettait justement d'exécuter et de faire beaucoup plus d'expérience à la fois.

Ceci m'a justement permis d'aller beaucoup plus vite. Et de faire? En tout 45 expériences. De tester 45 approches différentes en moins de 4 heures. Ce qui donne les normes m'aurait pris deux ou trois jours si j'avais utilisé que et ça c'est particulièrement si j'avais eu tout mon temps. Donc?

À la fin j'ai pu obtenir des résultats. Et je pense avoir décrire les principales difficultés que j'ai rencontrées dans ce document. La forme particulière, la principale contribution que j'aimerais apporter ici. C'est Ou du moins ce que j'ai le plus appris ici. C'est la façon justement d'optimiser les modèles. Étant donné que je me suis rendu compte que certains modèles n'étaient pas forcément.

Totalement compatible avec tensor touch compil que la grâce ou à cause de leur architecture j'ai opté pour les optimisations. Une approche d'optimisation. Par zone. Un modèle de machine peut être découpé en plusieurs en plusieurs blocs et l'idée était de regarder les blocs qui peuvent être optimisés. Je prends le cas de Retina et Retina a un backbone, c'est-à-dire une partie principale qui est faite avec qui est basé sur l'architecture de.

De r50. Donc? Ce que j'ai fait c'est que je pouvais optimiser uniquement cette partie là avec toshcompagne. Et les parties qui ont tendance à gêner je les optimise de différentes façons. Je peux utiliser par exemple amp. Pour je peux utiliser par exemple il s'est passé là je peux juste les passer en ftcs juste pour accélérer parce que je ne peux par exemple pas.

Il était question là. De diviser un modèle suivant les zones et pour chaque zone. Regardez les contraintes qu'il y a et appliquer la meilleure optimisation pour cette zone là. C'est une méthode qui m'a permis d'avoir des résultats assez intéressants également. Bien qu'il n'était pas forcément meilleur et je pense que si j'ai beaucoup plus de temps si j'avais eu beaucoup plus de temps, j'aurais pu travailler cela beaucoup plus encore.

Qui t'a par exemple pour les modèles qui ont des. Des parties spécifiques que ce soit le encore la partie qui? C'est à dire la que? De sélection de F. Crosse où il y a la sélection des encres ou même un MMS de Retina. Il aurait été possible soit de chercher un module MMS équivalent qui est écrit optimisé.

Pour le GPU ou alors de faire une réglementation optimiser pour le GPU? De cela? Quitte à écrire du code Triton. Parce que je ne suis peut-être pas en quoi écrire kuda.