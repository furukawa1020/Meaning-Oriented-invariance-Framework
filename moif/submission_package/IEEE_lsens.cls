%% IEEE_lsens.cls - IEEE Sensors Letters class file
%% Based on IEEEtran.cls with IEEE Sensors Letters specific settings
%% Compatible replacement for submissions to IEEE Sensors Letters
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IEEE_lsens}[2024/01/01 IEEE Sensors Letters]

% Pass all options to IEEEtran
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{IEEEtran}}
\ProcessOptions\relax

% Load IEEEtran with journal and twocolumn options
\LoadClass[journal,twocolumn,9pt]{IEEEtran}

% Required packages
\RequirePackage{cite}
\RequirePackage{amsmath,amssymb,amsfonts}
\RequirePackage{graphicx}
\RequirePackage{textcomp}
\RequirePackage{xcolor}

% Journal name for header
\def\@journalname{IEEE Sensors Letters}

% Set the journal header
\newcommand{\journalname}{IEEE SENSORS LETTERS}

% Override the header to show IEEE Sensors Letters
\markboth{\journalname}{\journalname}

% Subject Classification line support
\newcommand{\subjectC}[1]{\begin{center}\textit{Subject classification}: #1\end{center}}

% EDICS
\newcommand{\EDICS}[1]{\vspace{-0.5em}\begin{center}\small\textit{EDICS}: #1\end{center}}

\endinput
