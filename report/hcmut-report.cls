\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hcmut-report}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[12pt,a4paper]{article}

\RequirePackage[utf8]{vntex,inputenc}
\RequirePackage{anyfontsize,amsmath,amssymb}
\RequirePackage[T5]{fontenc}

\RequirePackage{geometry,graphicx,indentfirst,setspace}
\RequirePackage{tikz,etoolbox}
\RequirePackage{booktabs}
\RequirePackage[center]{caption}

\RequirePackage{fancyhdr,hyperref}
\RequirePackage[super,square,sort]{natbib}

% Change to '[english,vietnamese]' to use Vietnamese labels
\RequirePackage[english,vietnamese]{babel}

% Custom commands
\newcommand{\Month}{
  \ifcase \the\month
  \or January
  \or February
  \or March
  \or April
  \or May
  \or June
  \or July
  \or August
  \or September
  \or October
  \or November
  \or December
  \fi
}

\newcommand{\@upperuniname}{Vietnam National University Ho Chi Minh City}
\newcommand{\@uniname}{Ho Chi Minh City University of Technology}

\newcommand{\@deptname}{Faculty of Computer Science and Engineering}
\newcommand{\deptname}[1]{\renewcommand{\@deptname}{#1}}

\newcommand{\@coursename}{Course name}
\newcommand{\coursename}[1]{\renewcommand{\@coursename}{#1}}

\newcommand{\@reporttype}{Assignment Report}
\newcommand{\reporttype}[1]{\renewcommand{\@reporttype}{#1}}

\renewcommand{\@title}{Report title}

\newcommand{\@advisor}{Supervisor}
\newcommand{\advisor}[1]{\renewcommand{\@advisor}{#1}}

\newcommand{\@stuname}{STUDENT:\ H (xxxxxxx)}
\newcommand{\stuname}[1]{\renewcommand{\@stuname}{#1}}

% Cover metadata: dedicated fields, preserving older advisor/stuname commands.
\newcommand{\@major}{Computer Science}
\newcommand{\major}[1]{\renewcommand{\@major}{#1}}
\newcommand{\@supervisorone}{}
\newcommand{\supervisorone}[1]{\renewcommand{\@supervisorone}{#1}}
\newcommand{\@supervisortwo}{}
\newcommand{\supervisortwo}[1]{\renewcommand{\@supervisortwo}{#1}}
\newcommand{\@studentone}{}
\newcommand{\studentone}[1]{\renewcommand{\@studentone}{#1}}
\newcommand{\@studenttwo}{}
\newcommand{\studenttwo}[1]{\renewcommand{\@studenttwo}{#1}}
\newcommand{\@studentthree}{}
\newcommand{\studentthree}[1]{\renewcommand{\@studentthree}{#1}}
\newcommand{\@studentfour}{}
\newcommand{\studentfour}[1]{\renewcommand{\@studentfour}{#1}}

\newcommand{\@reportplace}{Ho Chi Minh City}
\newcommand{\reportplace}[1]{\renewcommand{\@reportplace}{#1}}

\newcommand{\@reporttime}{\Month \the\year}
\newcommand{\reporttime}[1]{\renewcommand{\@reporttime}{#1}}

% Configurations
%\graphicspath{{subdir1/}{subdir2/}{subdir3/}...{subdirn/}}

\geometry{
  paper=a4paper,
  left=3cm,
  right=2cm,
  vmargin=2cm,
  includeheadfoot=true,
  headheight=35pt
}

\onehalfspacing
\raggedbottom
\renewcommand{\arraystretch}{1.2}

% Set depth of numbering for sections and table of contents
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}
 \setlength{\headheight}{37.70004pt}
% Set depth of numbering for counters
\AtBeginDocument{\counterwithin{equation}{section}}
\AtBeginDocument{\counterwithin{table}{section}}
\AtBeginDocument{\counterwithin{figure}{section}}

\AtBeginDocument{\renewcommand*\bibname{Tài liệu tham khảo}}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=red,
  urlcolor=blue
}

\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0pt}

\pagestyle{fancy}
\fancyhf{}

%\renewcommand{\subsectionmark}[1]{\markright{\thesubsection.\ #1}}

% The drawing units are interpreted as multiples of \unitlength
% unless a unit is given
%\setlength{\unitlength}{1pt}

\if@twoside
  \fancyhead[EL]{
    \bfseries
    \begin{tabular}{rl}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{graphics/hcmut.png}}
      \end{picture}
      \begin{tabular}{l}
        \ttfamily\@uniname \\
        \ttfamily\@deptname \\
      \end{tabular}
    \end{tabular}
  }
  \fancyhead[OR]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{graphics/hcmut.png}}
      \end{picture}
    \end{tabular}
  }
\else
  \fancyhead[R]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{graphics/hcmut.png}}
      \end{picture}
    \end{tabular}
  }
\fi

\fancyfoot[C]{\thepage}

% Custom pages and sections
% Cover styled after HK251-DAGD1-337_DSA_Visual.pdf, page 1.
\newcommand{\coverpage}{%
  \clearpage
  \begingroup
  \thispagestyle{empty}%
  \newgeometry{top=2cm,bottom=2cm,left=2cm,right=2cm}%
  % Two nested borders are positioned relative to the paper, not the text area.
  \begin{tikzpicture}[remember picture,overlay]
    \draw[line width=1.8pt]
      ([xshift=1.02cm,yshift=-1.20cm]current page.north west)
      rectangle
      ([xshift=-1.02cm,yshift=1.20cm]current page.south east);
    \draw[line width=0.45pt]
      ([xshift=1.13cm,yshift=-1.31cm]current page.north west)
      rectangle
      ([xshift=-1.13cm,yshift=1.31cm]current page.south east);
  \end{tikzpicture}\par
  \centering
  \vspace*{-0.35cm}%
  {\fontsize{13}{17}\selectfont
    ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH\par
    TRƯỜNG ĐẠI HỌC BÁCH KHOA\par
    KHOA KHOA HỌC VÀ KỸ THUẬT MÁY TÍNH\par}
  \vspace{1.1cm}
  \includegraphics[width=3.25cm,keepaspectratio]{graphics/hcmut.png}\par
  \vspace{1.45cm}
  {\bfseries\fontsize{15}{18}\selectfont
    \@coursename\par}
  \vspace{1.35cm}
  {\bfseries\fontsize{18}{23}\selectfont
    \begin{minipage}{0.93\textwidth}\centering
      \@title
    \end{minipage}\par}
  \vspace{0.38cm}
  {\fontsize{12}{15}\selectfont Ngành: \@major\par}
  \vspace{1.15cm}
  {\fontsize{12}{16}\selectfont
    \begin{minipage}{0.84\textwidth}
      \raggedright
      \makebox[6.6cm][l]{\textbf{GIẢNG VIÊN HƯỚNG DẪN:}}\@supervisorone\par
      \ifdefempty{\@supervisortwo}{}{\makebox[6.6cm][l]{}\@supervisortwo\par}
      \centering\vspace{0.06cm}---o0o---\par
      \raggedright
      \textbf{SINH VIÊN THỰC HIỆN:}\par
      \hspace*{1.6cm}\@studentone\par
      \ifdefempty{\@studenttwo}{}{\hspace*{1.6cm}\@studenttwo\par}
      \ifdefempty{\@studentthree}{}{\hspace*{1.6cm}\@studentthree\par}
      \ifdefempty{\@studentfour}{}{\hspace*{1.6cm}\@studentfour\par}
    \end{minipage}\par}
  \vfill
  {\fontsize{12}{15}\selectfont \@reportplace, \@reporttime\par}
  \vspace*{-0.15cm}
  \restoregeometry
  \endgroup
  \clearpage
  \pagestyle{fancy}%
}
