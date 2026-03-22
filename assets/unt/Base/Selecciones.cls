\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Base/Selecciones}[2020/03/11 v1.0 Revista SELECCIONES MATEMATICAS]

\RequirePackage{ragged2e}  % Para ajustar la alineación del texto
\RequirePackage{xcolor}
	\DefineNamedColor{named}{FGreen}{cmyk}{0.91,0,0.88,0.12}
\RequirePackage{graphics}
%\RequirePackage{graphicx}
\RequirePackage{tikz}
\RequirePackage{epstopdf}
\RequirePackage{times}
\RequirePackage{fancyhdr}
	\pagestyle{fancy}

\DeclareOption{leqno}{\input{leqno.clo}}
\ProcessOptions

\setlength\paperwidth{210mm}  % Ancho del papel
\setlength\paperheight{297mm}  % Alto del papel
\@twosidetrue\@mparswitchtrue  % Papel a doble cara
\setlength\overfullrule{0pt}  % Regla final en 0pt
\input{Base/selec10.clo}  % Configuración del tamaño de letra
\setlength\lineskip{1\p@}  % Salto de línea
\setlength\normallineskip{1\p@}  % Salto de línea normal
\renewcommand\baselinestretch{}  % Tamaño de separación entre líneas
\setlength\parskip{0\p@ \@plus \p@}  % Salto de párrafo
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301

\setcounter{topnumber}{2}
\renewcommand\topfraction{.9}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.1}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.1}
\renewcommand\floatpagefraction{.91}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}

\renewcommand{\ps@plain}{
	\let\@mkboth\@gobbletwo%
	\renewcommand{\@oddhead}{}%
	\renewcommand{\@evenhead}{}%
	\renewcommand{\@evenfoot}{\hfil\footnotesize\textrm{\thepage}\hfil}%
	\renewcommand{\@oddfoot}{\@evenfoot}
}

\def\ps@headings{%
	\let\@oddfoot\@empty\let\@evenfoot\@empty
	\def\@evenhead{\rlap{\thepage}\hfil\upshape\footnotesize\leftmark\hfil}%          %!
	\def\@oddhead{\hbox{}\hfil{\upshape\footnotesize\rightmark}\hfil\llap{\thepage}}% %!
	\let\@mkboth\markboth
	
	\def\sectionmark##1{%
		\markboth{\uppercase{\ifnum \c@secnumdepth >\z@ %%%%\uppercase
				\thesection.\hskip 1em\relax\fi
				##1}}{}
	}%
	
	\def\subsectionmark##1{%
		\markright{\ifnum \c@secnumdepth >\@ne
			\thesubsection\hskip 1em\relax \fi
			##1}
	}
}

\def\ps@myheadings{%
	\let\@oddfoot\@empty\let\@evenfoot\@empty
	\def\@evenhead{\rlap{\thepage}\hfil\upshape\footnotesize\leftmark\hfil}%          %!
	\def\@oddhead{\hbox{}\hfil{\upshape\footnotesize\rightmark}\hfil\llap{\thepage}}% %!
	\let\@mkboth\@gobbletwo
	\let\sectionmark\@gobble
	\let\subsectionmark\@gobble
}

\newcommand\maketitle{
	\par
	\begingroup
		\renewcommand{\thefootnote}{\fnsymbol{footnote}}%
		\def\@makefnmark{\hbox to\z@{$\m@th^{\@thefnmark}$\hss}}%
		\long\def\@makefntext##1{\parindent 1em\noindent
			\hbox to1.8em{\hss$\m@th^{\@thefnmark}$}##1}%
		\global\@topnum\z@
		\@maketitle
		\thispagestyle{plain}\@thanks
	\endgroup
	\setcounter{footnote}{0}%
	\let\maketitle\relax\let\@maketitle\relax
	\gdef\@thanks{}\gdef\@author{}\gdef\@title{}\let\thanks\relax
}

\long\def\subtitle#1{\gdef\@subtitle{\par\vskip8pt\footnotesize#1}}
\def\@subtitle{}

\long\def\supertitle#1{\gdef\@supertitle{\par\bfseries#1\vskip17.5pt\par}}
\def\@supertitle{}


\gdef\@maketitle{%
	\justify  % Justificar el texto
	\setcounter{page}{\@startPage}  % Numeración de página
	%
	\begin{center}
		\textcolor{blue}{Journal homepage}\quad \@homepageLink\\
		\vskip 0.5em
		\begin{tikzpicture}
		\draw[outer color=green, inner color=white] (0, 2.7) rectangle (15, 2.6);
		\draw[path picture={\node at (path picture bounding box.center){\includegraphics[height=2.0cm]{Base/IMG/LOGOMS.eps}};},
		draw=green,very thick] (0, 2.5) rectangle (2.35, 0.2);
		\filldraw[fill=blue!10, draw=blue!10, thick] (2.45, 2.5) rectangle (12.65, 1.8) [path picture={
			\node at (path picture bounding box.center){\Large SELECCIONES MATEM\'ATICAS};}];
		\filldraw[fill=blue!10, draw=blue!10, thick] (2.45, 1.8) rectangle (12.65, 1.2) [path picture={
			\node at (path picture bounding box.center){\large Universidad Nacional de Trujillo};}];
		\filldraw[fill=blue!10, draw=blue!10, thick] (2.45, 1.2) rectangle (12.65, 0.7) [path picture={
			\node at (path picture bounding box.center){\large \@ISSN};}];
		\filldraw[fill=blue!10, draw=blue!10, thick] (2.45, 0.7) rectangle (12.65, 0.2) [path picture={
			\node at (path picture bounding box.center){\large \@edition.};}];
		\draw[path picture={\node at (path picture bounding box.center){\includegraphics[height=2.0cm]{Base/IMG/ssnnstat.eps}};},
		draw=green,very thick] (12.75, 2.5) rectangle (15, 0.2);
		\draw[outer color=green,inner color=white] (0, 0.1) rectangle (15,0);
		\end{tikzpicture}
		%
		\vskip 0.5em
		%
		\ifpreview
		\hfill {\Large REVIEW}
		\else
		\ifcommunication
		\hfill {\Large COMMUNICATION}
		\else
		\ifeducation
		\hfill {\Large MATHEMATICS EDUCATION}
		\fi
		\fi
		\fi
		
		\author{\@authors}
		\par
		\null
		{\bfseries\@ucnothanks\@title\par}%
		\vskip .075in%
		{\footnotesize
			\spaceskip=1.5\fontdimen2\the\font
			\def\and{{\, \scriptsize and \,\,}}
			\@tempswafalse\expandafter\@doauth\@author\and\relax}%
		
		\vspace{0.7em}
		
		{\it Received, \@received \hspace{2.3cm} Accepted, \@accepted}\hspace{3cm}  \href{https://creativecommons.org/licenses/by/4.0/}{\includegraphics[scale=0.5]{Base/IMG/cc-by.png}}
	\end{center}
	
	\vspace{0.3cm}
	
	\noindent\textcolor{blue}{How to cite this article:}\\
	{\small \@howtocite \hspace{1em} }
	
	\begin{center}
		\noindent \textcolor{FGreen}{\rule{15cm}{0.6mm}}
	\end{center}
	\par
	\vskip .11in
}

%%% Comandos y variables
%==================== MACROS ====================%
\def\ID#1{\href{#1}{\includegraphics[scale=0.055]{Base/IMG/indice1.eps}}}

\def\information{
	\par
	\bigskip
	\noindent \textcolor{FGreen}{{\bf ORCID and License}}\\
	%
	\loop
	\stepcounter{infocount}
	%
	\getautor{\theinfocount} \url{\getorcid{\theinfocount}}\\
	\ifnum \value{infocount}<\theautcount
	\repeat
	
	\noindent This work is licensed under the  \href{https://creativecommons.org/licenses/by/4.0/}{Creative Commons -  Attribution 4.0 International (CC BY 4.0)} 
}

\def\marginsize#1#2#3#4{
	\hoffset = -1in  % Le quitamos el espacio antes del margen izquierdo
	\voffset = -1in  % Le quitamos el espacio antes del margen superior
	\oddsidemargin = #1  % Margen izquierdo para páginas impares
	\evensidemargin = #1  % Margen derecho para páginas pares
	\headheight = 5mm  % Altura de la cabecera
	\headsep = 3mm  % Separación de la cabecera y cuerpo del texto
	\topmargin = #3  % Margen superior
	\textwidth = \dimexpr\paperwidth-\oddsidemargin-#2\relax   % Ancho de la zona de texto
	\textheight = \dimexpr\paperheight-\topmargin-#4-\headsep-\headheight\relax  % Alto de la zona de texto
	\footskip = 10mm  % Separación entre el texto y pie de página
}

\def\edition#1#2#3#4{
	\gdef\@year{#1}
	\gdef\@vol{#2}
	\gdef\@startPage{#3}
	\gdef\@endPage{#4}
	\gdef\@edition{\@year; Vol. \@vol: \@startPage-\@endPage}
}

\def\autor#1#2#3{
	\stepcounter{autcount}
	\expandafter\def\csname autor\theautcount\endcsname{#1}
	\expandafter\def\csname orcid\theautcount\endcsname{#2}
	\expandafter\def\csname thx\theautcount\endcsname{\thanks{#3}}
}

\def\getautor#1{\csname autor#1\endcsname}
\def\getorcid#1{\csname orcid#1\endcsname}
\def\getthx#1{\csname thx#1\endcsname}

\def\@authors{
	\loop
	\stepcounter{itercount}
	%
	{\getautor{\theitercount}\getthx{\theitercount}\ID{\getorcid{\theitercount}}}
	%
	\ifnum \value{itercount}< \numexpr \theautcount -1\relax
	\cand
	\else
	\ifnum \value{itercount}< \theautcount
	\and
	\fi
	\fi
	\ifnum \value{itercount}<\theautcount
	\repeat
}

\def\homepageLink#1{\gdef\@homepageLink{\url{#1}}}

\def\ISSN#1{\gdef\@ISSN{ISSN: #1}}

\def\doi#1{\gdef\@doi{{\bf\url{#1}}}}

\def\received#1{\gdef\@received{#1}}

\def\accepted#1{\gdef\@accepted{#1}}

\def\howtocite#1#2{
	\gdef\@journalmark{#1}
	\gdef\@howtocite{#1 \emph{#2}. Selecciones Matem\'aticas. \@year;\@vol:\@startPage--\@endPage. \hspace{0.2em} \@doi}
}

\def\headertext{
	\pagestyle{myheadings}
	\thispagestyle{plain}
	\markboth{\@journalmark - Selecciones Matem\'aticas. \@edition}
	{\@journalmark - Selecciones Matem\'aticas. \@edition}
}

\def\cand{{\kern-1mm ,}}

\def\@ucnothanks#1{\expandafter\@ucnt#1\thanks\relax}
\def\@ucnt#1\thanks{\large {#1}\futurelet\@tempa\@ucnta}
\def\@ucnta{\ifx\@tempa\relax\else\@ucntb\fi}
\def\@ucntb#1\thanks\relax{\thanks{#1}}
\def\@doauth#1\and{\let\@nextdoa=\@doauth
	\def\@tempc{\@ucnothanks{#1}}
	\futurelet\@tempa\@doautha
}
\def\@doautha{
	\ifx\@tempa\relax
		\if@tempswa {\and}\fi
		\let\@nextdoa=\relax
	\else
		\if@tempswa \gdef\and{\ \scriptsize and }, \fi   %%%%%%%  AND
	\fi
	\@tempc\unskip\@tempswatrue\@nextdoa
}

%\newcommand{\Rn}{\mathbb{R}^n}  % Símbolo del espacio real n-dimensional
\newcommand{\Rn}[1]{\mathbb{R}^{#1}}  % Símbolo de espacio real de una dimensión ingresada
\newcommand{\R}{\mathbb{R}}  % Símbolo del campo de los números reales
\newcommand{\Z}{\mathbb{Z}}  % Símbolo del anillo de los números enteros
\newcommand{\N}{\mathbb{N}}  % Símbolo del monoide los números naturales
\newcommand{\Q}{\mathbb{Q}}  % Símbolo del campo de los números racionales
\newcommand{\C}{\mathbb{C}}  % Símbolo del campo de los números racionales
% some definitions of bold math italics to make typing easier.
% They are used in the corollary.
%\def\bfE{\mbox{\boldmath$E$}}
%\def\bfG{\mbox{\boldmath$G$}}
\def\squareproof{$\,\displaystyle\square$}%\blacksquare
\newcommand{\fin}{\hfill$\square$}
%==================== CONDICIONALES ====================%
\newif\ifpreview
\newif\ifcommunication
\newif\ifeducation

%==================== CONTADORES ====================%
\newcounter{autcount}
\newcounter{itercount}
\newcounter{infocount}


%==================== VARIABLES ====================%
\def\@homepageLink{\url{http://revistas.unitru.edu.pe/index.php/SSMM}}
\def\@ISSN{ISSN: 2411-1783 (Online)}
\def\@received{}
\def\@accepted{}
\def\@doi{}
\def\@year{}
\def\@vol{}
\def\@startPage{}
\def\@endPage{}
\def\@edition{}
\def\@howtocite{}
\def\@journalmark{}

\let\and=\relax

\setcounter{secnumdepth}{3}
\newcounter {section}
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand\thesection       {\arabic{section}}
\renewcommand\thesubsection    {\thesection.\arabic{subsection}}
\renewcommand\thesubsubsection {\thesubsection.\arabic{subsubsection}}
\renewcommand\theparagraph     {\thesubsubsection.\arabic{paragraph}}
\renewcommand\thesubparagraph  {\theparagraph.\arabic{subparagraph}}


\def\@sect#1#2#3#4#5#6[#7]#8{\ifnum #2>\c@secnumdepth
	\def\@svsec{}\else
	\refstepcounter{#1}\edef\@svsec{\csname the#1\endcsname.\hskip .5em }\fi
	\@tempskipa #5\relax
	\ifdim \@tempskipa>\z@
	\begingroup #6\relax
	\@hangfrom{\hskip #3\relax\@svsec}{\interlinepenalty \@M #8\par}
	\endgroup
	\csname #1mark\endcsname{#7}\addcontentsline
	{toc}{#1}{\ifnum #2>\c@secnumdepth \else
		\protect\numberline{\csname the#1\endcsname}\fi
		#7}\else
	\def\@svsechd{#6\hskip #3\@svsec #8.\csname #1mark\endcsname
		{#7}\addcontentsline
		{toc}{#1}{\ifnum #2>\c@secnumdepth \else
			\protect\numberline{\csname the#1\endcsname}\fi
			#7}}\fi
	\@xsect{#5}}
\def\@ssect#1#2#3#4#5{\@tempskipa #3\relax
	\ifdim \@tempskipa>\z@
	\begingroup #4\@hangfrom{\hskip #1}{\interlinepenalty \@M
		#5.\par}\endgroup
	\else \def\@svsechd{#4\hskip #1\relax #5.}\fi
	\@xsect{#3}}


\newcommand\section{\@startsection{section}{1}{.25in}%
	{1.3ex \@plus .5ex \@minus .2ex}%
	{-.5em \@plus -.1em}%
	{\reset@font\normalsize\bfseries}}
\newcommand\subsection{\@startsection{subsection}{2}{.25in}%
	{1.3ex\@plus .5ex \@minus .2ex}%
	{-.5em \@plus -.1em}%
	{\reset@font\normalsize\bfseries}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{.25in}%
	{1.3ex\@plus .5ex \@minus .2ex}%
	{-.5em \@plus -.1em}%
	{\reset@font\normalsize\bfseries}}
\newcommand\paragraph{\@startsection{paragraph}{4}{.25in}%
	{\parskip}%
	{-.5em plus -.1em}%
	{\reset@font\normalsize\itshape}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
	{\parskip}%
	{-.5em plus -.1em}%
	{\reset@font\normalsize\itshape}}
\if@twocolumn
\setlength\leftmargini  {2em}
\else
\setlength\leftmargini  {0.5in}
\fi
\setlength\leftmarginii  {22pt}
\setlength\leftmarginiii {18.7pt}
\setlength\leftmarginiv  {17pt}
\if@twocolumn
\setlength\leftmarginv  {.5em}
\setlength\leftmarginvi {.5em}
\else
\setlength\leftmarginv  {10pt}
\setlength\leftmarginvi {10pt}
\fi
\setlength  \leftmargin{\leftmargini}
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\arabic{enumi}}
\renewcommand\theenumii{\alph{enumii}}
\renewcommand\theenumiii{\roman{enumiii}}
\renewcommand\theenumiv{\Alph{enumiv}}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi\theenumii}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{$\m@th\bullet$}
\newcommand\labelitemii{\normalfont\bfseries --}
\newcommand\labelitemiii{$\m@th\ast$}
\newcommand\labelitemiv{$\m@th\cdot$}
\newenvironment{description}
{\list{}{\labelwidth\z@ \itemindent-\leftmargin
		\let\makelabel\descriptionlabel}}
{\endlist}
\newcommand\descriptionlabel[1]{\hspace\labelsep
	\normalfont\bfseries #1}
\newcounter{rmnum}
\newenvironment{romannum}
{\begin{list}{{\upshape (\roman{rmnum})}}{\usecounter{rmnum}
			\setlength{\leftmargin}{0pt}
			\setlength{\itemindent}{42pt}}}{\end{list}}
\newcounter{muni}
\newenvironment{remunerate}
{\begin{list}{{\upshape \arabic{muni}.}}{\usecounter{muni}
			\setlength{\leftmargin}{0pt}
			\setlength{\itemindent}{38pt}}}{\end{list}}

\newenvironment{@abssec}[1]{%
	\if@twocolumn
	\section*{#1}%
	\else
	\vspace{.05in}\footnotesize
	\parindent .2in
	{\upshape\bfseries #1. }\ignorespaces
	\fi}
{\if@twocolumn\else\par\vspace{.1in}\fi}

\newenvironment{@doisec}[1]{%
	\if@twocolumn
	\section*{#1}%
	\else
	\vspace{.05in}\footnotesize
	\parindent .2in
	{\upshape\bfseries #1. }\ignorespaces\ignorespaces
	\fi}
{\if@twocolumn\else\par\vspace{.1in}\fi}

%\newenvironment{abstract}{\begin{@abssec}{\abstractname}}{\end{@abssec}}
\newenvironment{abstract}
{\begin{center}{\bf\abstractname}\end{center}\hspace{-4pt}\it\noindent}{}
\newenvironment{keywords}{\small{\bf\keywordsname}.}{\par\null}
\newenvironment{AMS}{\begin{@abssec}{\AMSname}}{\end{@abssec}}
\newenvironment{AM}{\begin{@abssec}{\AMname}}{\end{@abssec}}
%\newenvironment{PII}{\begin{@abssec}{\PIIname}}{\end{@abssec}}
\newenvironment{DOI}{\begin{@doisec}{\DOIname}}{\end{@doisec}}
\newenvironment{introduction}{\section{\introductionname}}{}



\newenvironment{verse}
{\let\\=\@centercr
	\list{}{\itemsep      \z@
		\itemindent   -1.5em%
		\listparindent\itemindent
		\rightmargin  \leftmargin
		\advance\leftmargin 1.5em}%
	\item[]}
{\endlist}
\newenvironment{quotation}
{\list{}{\listparindent 1.5em%
		\itemindent    \listparindent
		\rightmargin   \leftmargin
		\parsep        \z@ \@plus\p@}%
	\item[]}
{\endlist}
\newenvironment{quote}
{\list{}{\rightmargin\leftmargin}%
	\item[]}
{\endlist}
\def\@begintheorem#1#2{\par\bgroup{\bf
		#1\ #2. }\it\ignorespaces}
\def\@opargbegintheorem#1#2#3{\par\bgroup%
	{\bf 
		#1\ #2\ ({\upshape #3}). }\it\ignorespaces}
\def\@endtheorem{\egroup}
%\def\proof{\par{\it Proof}. \ignorespaces}

%\def\endproof{\vbox{\hrule height0.6pt\hbox{%
%   \vrule height1.3ex width0.6pt\hskip0.8ex
%   \vrule width0.6pt}\hrule height0.6pt
%  }}

\newtheorem{theorem}{\theoremname}[section]
\newtheorem{lemma}{\lemmaname}[section]
\newtheorem{corollary}{\corollaryname}[section]
\newtheorem{proposition}{\propositionname}[section]
\newtheorem{definition}{\definitionname}[section]
\newtheorem{concept}{\conceptname}[section]
\newtheorem{example}{\examplename}[section]
\newtheorem{commentary}{\commentaryname}[section]
\newtheorem{remark}{\obsname}[section]
\newtheorem{note}{\notename}[section]

\newtheorem{theorem*}{\theoremname}
\newtheorem{lemma*}{\lemmaname}
\newtheorem{corollary*}{\corollaryname}
\newtheorem{proposition*}{\propositionname}
\newtheorem{definition*}{\definitionname}
\newtheorem{concept*}{\conceptname}
\newtheorem{example*}{\examplename}
\newtheorem{commentary*}{\commentaryname}
\newtheorem{remark*}{\obsname}
\newtheorem{note*}{\notename}

\newcommand{\proof}{{\it\proofname: }}

% FROM FIXUP.STY
% We now redefine the eqnarray environment to correct the space around
% the operator.   Any changes to the definition in LaTeX itself
% will cause the `Warning: unable...'
% message to appear.  If this happens it will be necessary to update the
% definition of  \@tempb below and incorperate the changes into
% the new definitions of \eqnarray.


% This is the current definition of \eqnarray for the fleqn document style
% latex.ltx

\def\@tempb{%
	\stepcounter{equation}%
	\def\@currentlabel{\p@equation\theequation}%
	\global\@eqnswtrue
	\m@th
	\global\@eqcnt\z@
	\tabskip\@centering
	\let\\\@eqncr
	$$\everycr{}\halign to\displaywidth\bgroup
	\hskip\@centering$\displaystyle\tabskip\z@skip{##}$\@eqnsel
	&\global\@eqcnt\@ne\hskip \tw@\arraycolsep \hfil${##}$\hfil
	&\global\@eqcnt\tw@ \hskip \tw@\arraycolsep
	$\displaystyle{##}$\hfil\tabskip\@centering
	&\global\@eqcnt\thr@@ \hb@xt@\z@\bgroup\hss##\egroup
	\tabskip\z@skip
	\cr
}
\ifx\eqnarray\@tempb    % Try the default eqnarray environment
\def\eqnarray{%
	\stepcounter{equation}%
	\def\@currentlabel{\p@equation\theequation}%
	\global\@eqnswtrue
	\m@th
	\global\@eqcnt\z@
	\tabskip\@centering
	\let\\\@eqncr
	$$\everycr{}\halign to\displaywidth\bgroup
	\hskip\@centering$\displaystyle\tabskip\z@skip{##}$\@eqnsel
	&\global\@eqcnt\@ne \hfil$\displaystyle{{}##{}}$\hfil
	&\global\@eqcnt\tw@ $\displaystyle{##}$\hfil\tabskip\@centering
	&\global\@eqcnt\thr@@ \hb@xt@\z@\bgroup\hss##\egroup
	\tabskip\z@skip
	\cr
}
\else    \typeout{Warning: Unable to fix unknown version of \string\eqnarray.}
\fi

\def\@tempb{}


%% New Mathops.

\newcommand\const{\mathop{\operator@font const}\nolimits}
\newcommand\diag{\mathop{\operator@font diag}\nolimits}
\newcommand\grad{\mathop{\operator@font grad}\nolimits}
\newcommand\Range{\mathop{\operator@font Range}\nolimits}
\newcommand\rank{\mathop{\operator@font rank}\nolimits}
\newcommand\supp{\mathop{\operator@font supp}\nolimits}

\newcommand\appendix{%
	\renewcommand{\section}{%
		\secdef\append\sappend}%
	\setcounter{section}{0}%
	\setcounter{subsection}{0}%
	\renewcommand\thesection{\Alph{section}}}

\newcommand\Appendix{%
	\renewcommand{\section}{%
		\secdef\Append\sAppend}%
	\setcounter{section}{0}%
	\setcounter{subsection}{0}%
	\renewcommand\thesection{\Alph{section}}}

\newcommand\append[2][?]{%
	\refstepcounter{section}%
	{\vskip 1.3ex \@plus .5ex \@minus .2ex
		\reset@font\normalsize\bfseries\appendixname\ \thesection. #2.}}

\newcommand\sappend[1]{%
	\refstepcounter{section}
	{\vskip 1.3ex \@plus .5ex \@minus .2ex
		\reset@font\normalsize\bfseries\appendixname\ \thesection.}}

\newcommand\Append[2][?]{%
	\refstepcounter{section}%
	{\vskip 1.3ex \@plus .5ex \@minus .2ex
		\reset@font\normalsize\bfseries\appendixname. #2.}}

\newcommand\sAppend[1]{%
	\refstepcounter{section}
	{\vskip 1.3ex \@plus .5ex \@minus .2ex
		\reset@font\normalsize\bfseries\appendixname.}}

\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}


\newcounter{figure}
\renewcommand\thefigure{\@arabic\c@figure}
\def\fps@figure{tp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
{\@float{figure}}
{\end@float}
\newenvironment{figure*}
{\@dblfloat{figure}}
{\end@dblfloat}
\newcounter{table}
\renewcommand{\thetable}{\@arabic\c@table}
\def\fps@table{tp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
{\@float{table}}
{\end@float}
\newenvironment{table*}
{\@dblfloat{table}}
{\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{10\p@}
\def\@figtxt{figure}
\long\def\@makecaption#1#2{%
	\footnotesize
	\setlength{\parindent}{1.5pc}
	\ifx\@captype\@figtxt
	\vskip\abovecaptionskip
	\setbox\@tempboxa\hbox{{\normalfont\scshape #1}. {\normalfont\itshape #2}}%
	\ifdim \wd\@tempboxa >\hsize
	{\normalfont\scshape #1}. {\normalfont\itshape #2}\par
	\else
	\global\@minipagefalse
	\hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
	\fi
	\else
	\hbox to\hsize{\hfil{\normalfont\scshape #1}\hfil}%
	\setbox\@tempboxa\hbox{{\normalfont\itshape #2}}%
	\ifdim \wd\@tempboxa >\hsize
	{\normalfont\itshape #2}\par
	\else
	\global\@minipagefalse
	\hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
	\fi
	\vskip\belowcaptionskip
	\fi}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch{\relax}{\mathcal}}
\DeclareRobustCommand*\mit{\@fontswitch{\relax}{\mathnormal}}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg {2.55em}
\newcommand\@dotsep{4.5}

\newdimen\bibindent
\bibindent=16pt
\newenvironment{thebibliography}[1]
{\par\addvspace{.25in}
	\begin{center}
		%\footnotesize\expandafter{\refname} % \footnotesize\uppercase\expandafter{\refname}
		\Large\expandafter{\bf\refname}
	\end{center}
	\@mkboth{\uppercase\expandafter{\refname}}{\uppercase\expandafter{\refname}}
	\addvspace{.15in}\nopagebreak%
	\list{\@biblabel{\arabic{enumi}}}%
	{\settowidth\labelwidth{\@biblabel{#1}}%
		\leftmargin\labelwidth
		\advance\leftmargin\labelsep
		%            \if@openbib
		\advance\leftmargin\bibindent
		\itemindent -\bibindent
		\listparindent \itemindent
		\parsep \z@
		%            \fi
		\usecounter{enumi}%
		\let\p@enumi\@empty
		\renewcommand\theenumi{\arabic{enumi}}}%
	%      \if@openbib
	%        \renewcommand\newblock{\par}
	%      \else
	\renewcommand\newblock{\hskip .11em \@plus.33em \@minus.07em}%
	%      \fi
	\sloppy\clubpenalty4000\widowpenalty4000%
	\sfcode`\.=\@m\relax\footnotesize}
{\def\@noitemerr
	{\@latex@warning{Empty `thebibliography' environment}}%
	\endlist}
\newcommand\newblock{}
\newenvironment{theindex}
{\if@twocolumn
	\@restonecolfalse
	\else
	\@restonecoltrue
	\fi
	\columnseprule \z@
	\columnsep 35\p@
	\twocolumn[\section*{\indexname}]%
	\@mkboth{\uppercase{\indexname}}%
	{\uppercase{\indexname}}%
	\thispagestyle{plain}\parindent\z@
	\parskip\z@ \@plus .3\p@\relax
	\let\item\@idxitem}
{\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem  {\par\hangindent 40\p@}
\newcommand\subitem   {\par\hangindent 40\p@ \hspace*{20\p@}}
\newcommand\subsubitem{\par\hangindent 40\p@ \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
	\kern-3\p@
	\hrule width .5in
	\kern 2.6\p@}
\long\def\@makefntext#1{%
	\parindent .25in%
	\noindent
	\hbox to .25in{\hss\@makefnmark}#1}
\newcommand\contentsname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Contents}\else{{\'I}ndice}\fi}
\newcommand\obsname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Remark}\else{\ifnum\pdfstrcmp{\languagename}{brazil}=0 {Obsderva\c c\~ao}\else{Observaci{\'o}n}\fi}\fi}
\newcommand\listfigurename{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {List of Figures}\else{Lista de Figuras}\fi}
\newcommand\listtablename{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {List of Tables}\else{\ifnum\pdfstrcmp{\languagename}{brazil}=0 {Lista de Tabelas}\else{Lista de Tablas}\fi}\fi}
\newcommand\refname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {References}\else{Bibliograf{\'}a}\fi}
\newcommand\indexname{Index}
\newcommand\figurename{Fig.}
\newcommand\tablename{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Table}\else{ifnum\pdfstrcmp{\languagename}{brazil}=0 {Tabela}\else{Tabla}\fi}\fi}
\newcommand\partname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Part}\else{Parte}\fi}
\newcommand\appendixname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Appendix}\else{\ifnum\pdfstrcmp{\languagename}{brazil}=0{Ap{\^}endice}\else{Ap{\'}ndice}\fi}\fi}
\newcommand\abstractname{
	\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Abstract}
	\else{\ifnum\pdfstrcmp{\languagename}{brazil}=0 {Resumo}\else{Resumen}\fi}\fi
}

\newcommand\keywordsname{
	\ifnum\pdfstrcmp{\languagename}{brazil}=0{Palavras-chave}
	\else{\ifnum\pdfstrcmp{\languagename}{USenglish}=0{Keywords}\else{Palabras clave}\fi}\fi
}
\newcommand\notename{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Note}\else{Nota}\fi}
%
%
\newcommand\theoremname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Theorem}\else{Teorema}\fi}
\newcommand\lemmaname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Lemma}\else{Lema}\fi}
\newcommand\corollaryname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Corollary}\else{Corolario}\fi}
\newcommand\propositionname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Proposition}\else{\ifnum\pdfstrcmp{\languagename}{brazil}=0 {Proposi\c c\~ao}\else{Proposici{\'o}n}\fi}\fi}
\newcommand\definitionname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Definition}\else{\ifnum\pdfstrcmp{\languagename}{brazil}=0 {Defini\c c\~ao}\else{Definici{\'o}n}\fi}\fi}
\newcommand\conceptname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Concept}\else{Concepto}\fi}
\newcommand\examplename{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Example}\else{\ifnum\pdfstrcmp{\languagename}{brazil}=0{Example}\else{Ejemplo}\fi}\fi}
\newcommand\commentaryname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Commentary}\else{\ifnum\pdfstrcmp{\languagename}{brazil}=0 {Comentário}\else{Comentario}\fi}\fi}
\newcommand\proofname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Proof}\else{\ifnum\pdfstrcmp{\languagename}{brazil}=0{Prova}\else{Prueba}\fi}\fi}
\newcommand\introductionname{\ifnum\pdfstrcmp{\languagename}{USenglish}=0 {Introduction}\else{\ifnum\pdfstrcmp{\languagename}{brazil}=0{Introdu\c c\~ao}\else{Introducc{\'i}on}\fi}\fi}
%
%
\newcommand\AMSname{AMS subject classifications}
\newcommand\AMname{AMS subject classification}
%\newcommand\PIIname{PII}
\newcommand\DOIname{DOI}
\newcommand\URL{\protect\\ \hspace*{15.37pt}http://www.siam.org/journals/}
\newcommand\today{\ifcase\month\or
	January\or February\or March\or April\or May\or June\or
	July\or August\or September\or October\or November\or December\fi
	\space\number\day, \number\year}
\newcommand\sameauthor{\leavevmode\vrule height 2pt depth -1.6pt width 23pt}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{plain}
\pagenumbering{arabic}      % Arabic page numbers

\if@compatibility%
\else
\@addtoreset{equation}{section}   %  \section resets 'equation' counter
\renewcommand{\theequation}{\thesection.\arabic{equation}}  
%

\@addtoreset{figure}{section}
\renewcommand{\thefigure}{\thesection.\@arabic\c@figure}
%
\@addtoreset{table}{section}
\renewcommand{\thetable}{\thesection.\@arabic\c@table}
%
\@addtoreset{theorem}{section}
\renewcommand{\thetheorem}{\thesection.\arabic{theorem}}
\fi

\onecolumn

\gdef\@siamlabel@error#1#2{%
	\GenericError{%
		\space\space\space\@spaces\@spaces\@spaces
	}{%
		SIAM Label Error: #1%
	}{%
		Correct multiply-defined labels.%
	}{#2}%
}

%%% NEW -- causes multiply defined labels to generate errors
\def\@newl@bel#1#2#3{%
	\@ifundefined{#1@#2}%
	\relax
	{\gdef \@multiplelabels {%
			\@latex@warning@no@line{Label `#2' multiply defined}}%
		\@siamlabel@error{There were multiply-defined labels}{}}%
	\global\@namedef{#1@#2}{#3}}
\def\newlabel{\@newl@bel r}
\@onlypreamble\@newl@bel
\let \@multiplelabels \relax
\def\label#1{\@bsphack
	\protected@write\@auxout{}%
	{\string\newlabel{#1}{{\@currentlabel}{\thepage}}}%
	\@esphack}
\def\refstepcounter#1{\stepcounter{#1}%
	\protected@edef\@currentlabel
	{\csname p@#1\endcsname\csname the#1\endcsname}%
}

\renewcommand{\arraystretch}{1.5}

\RequirePackage[colorlinks=true,citecolor=red,linkcolor=blue,urlcolor=cyan,filecolor=magenta]{hyperref}

\endinput