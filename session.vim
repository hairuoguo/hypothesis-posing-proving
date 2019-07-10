let SessionLoad = 1
if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
map! <D-v> *
nnoremap 	 >>
nnoremap <NL> :bp
nnoremap  :bn
nnoremap <silent>  :call append(line("."),   repeat([""], v:count1))
nnoremap  :call HLLine(0.1)
map  <Plug>(ctrlp)
nnoremap  :bp|bd #
nnoremap  :wa
nmap  + <Plug>BufTabLine.Go(12)
nmap  - <Plug>BufTabLine.Go(11)
nmap  0 <Plug>BufTabLine.Go(10)
nmap  9 <Plug>BufTabLine.Go(9)
nmap  8 <Plug>BufTabLine.Go(8)
nmap  7 <Plug>BufTabLine.Go(7)
nmap  6 <Plug>BufTabLine.Go(6)
nmap  5 <Plug>BufTabLine.Go(5)
nmap  4 <Plug>BufTabLine.Go(4)
nmap  3 <Plug>BufTabLine.Go(3)
nmap  2 <Plug>BufTabLine.Go(2)
nmap  1 <Plug>BufTabLine.Go(1)
noremap  u :s/#/:noh
noremap  p oprint('pa: ' + str(pa))
noremap  r @@
noremap  le i\begin{enumerate}[label=(\alph*)]oI\itemoI\end{enumerate}kA 
noremap  i k0yf(j^hv0pv0:s/\%V./ /g:noh
noremap  d 0d$
noremap  c 0i#
nnoremap <silent>   :call append(line(".")-1, repeat([""], v:count1))
noremap : ;
noremap ; :
vnoremap < <gv
vnoremap > >gv
noremap K "*
noremap S $
map Y y$
vmap gx <Plug>NetrwBrowseXVis
nmap gx <Plug>NetrwBrowseX
noremap gm :so $MYVIMRC
noremap <silent> gh :TagbarOpen fj:set relativenumber
noremap <silent> gH :TagbarToggle
noremap gb :e $MYVIMRC
noremap gdd g^dg$
noremap gD dg$
noremap gcc g^cg$ " note this will make gc (currently mapped) go slower
noremap gC cg$
map gyy g^yg$
map gY yg$ " has to be recursive so that y calls highlight text
noremap gs g$
noremap s _
vnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#BrowseX(expand((exists("g:netrw_gx")? g:netrw_gx : '<cfile>')),netrw#CheckIfRemote())
noremap <silent> <Plug>BufTabLine.Go(-1) :exe 'b'.get(buftabline#user_buffers(),-1,'')
noremap <silent> <Plug>BufTabLine.Go(13) :exe 'b'.get(buftabline#user_buffers(),12,'')
noremap <silent> <Plug>BufTabLine.Go(12) :exe 'b'.get(buftabline#user_buffers(),11,'')
noremap <silent> <Plug>BufTabLine.Go(11) :exe 'b'.get(buftabline#user_buffers(),10,'')
noremap <silent> <Plug>BufTabLine.Go(10) :exe 'b'.get(buftabline#user_buffers(),9,'')
noremap <silent> <Plug>BufTabLine.Go(9) :exe 'b'.get(buftabline#user_buffers(),8,'')
noremap <silent> <Plug>BufTabLine.Go(8) :exe 'b'.get(buftabline#user_buffers(),7,'')
noremap <silent> <Plug>BufTabLine.Go(7) :exe 'b'.get(buftabline#user_buffers(),6,'')
noremap <silent> <Plug>BufTabLine.Go(6) :exe 'b'.get(buftabline#user_buffers(),5,'')
noremap <silent> <Plug>BufTabLine.Go(5) :exe 'b'.get(buftabline#user_buffers(),4,'')
noremap <silent> <Plug>BufTabLine.Go(4) :exe 'b'.get(buftabline#user_buffers(),3,'')
noremap <silent> <Plug>BufTabLine.Go(3) :exe 'b'.get(buftabline#user_buffers(),2,'')
noremap <silent> <Plug>BufTabLine.Go(2) :exe 'b'.get(buftabline#user_buffers(),1,'')
noremap <silent> <Plug>BufTabLine.Go(1) :exe 'b'.get(buftabline#user_buffers(),0,'')
nnoremap <silent> <Plug>(ctrlp) :CtrlP
nnoremap <BS> <<
vmap <BS> "-d
vmap <D-x> "*d
vmap <D-c> "*y
vmap <D-v> "-d"*P
nmap <D-v> "*P
inoremap  :wa
let &cpo=s:cpo_save
unlet s:cpo_save
set background=dark
set backspace=indent,eol,start
set expandtab
set fileencodings=ucs-bom,utf-8,default,latin1
set formatoptions=cqt
set helplang=en
set hidden
set laststatus=2
set listchars=tab:>-
set mouse=a
set runtimepath=~/.vim,~/.vim/bundle/ctrlp.vim,~/.vim/bundle/tagbar,~/.vim/bundle/vim-buftabline,~/.vim/bundle/vim-highlightedyank,/usr/local/share/vim/vimfiles,/usr/local/share/vim/vim81,/usr/local/share/vim/vimfiles/after,~/.vim/after
set shiftwidth=4
set noshowmode
set showtabline=2
set softtabstop=4
set tabline=%!buftabline#render()
set tabstop=4
set textwidth=80
set ttimeoutlen=0
set visualbell
set wildignore=*.pyc
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Python/hypothesis-posing-proving
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +1 dqn/dqn_her.py
badd +1 deep_rl/agents/Base_Agent.py
badd +1 deep_rl/agents/Trainer.py
badd +1 deep_rl/nn_builder/pytorch/Base_Network.py
badd +1 deep_rl/nn_builder/pytorch/CNN.py
badd +1 deep_rl/nn_builder/pytorch/NN.py
badd +0 deep_rl/nn_builder/pytorch/cnn2.py
argglobal
silent! argdel *
$argadd dqn/dqn_her.py
$argadd deep_rl/agents/Base_Agent.py
$argadd deep_rl/agents/Trainer.py
$argadd deep_rl/nn_builder/pytorch/Base_Network.py
$argadd deep_rl/nn_builder/pytorch/CNN.py
$argadd deep_rl/nn_builder/pytorch/NN.py
$argadd deep_rl/nn_builder/pytorch/RNN.py
$argadd deep_rl/nn_builder/pytorch/__init__.py
$argadd deep_rl/nn_builder/pytorch/cnn2.py
edit deep_rl/nn_builder/pytorch/NN.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
if bufexists('deep_rl/nn_builder/pytorch/NN.py') | buffer deep_rl/nn_builder/pytorch/NN.py | else | edit deep_rl/nn_builder/pytorch/NN.py | endif
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal nocindent
setlocal cinkeys=0{,0},0),:,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
set colorcolumn=80
setlocal colorcolumn=80
setlocal comments=b:#,fb:-
setlocal commentstring=#\ %s
setlocal complete=.,w,b,u,t,i
setlocal concealcursor=
setlocal conceallevel=0
setlocal completefunc=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
setlocal cursorline
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'python'
setlocal filetype=python
endif
setlocal fixendofline
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=cqt
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal formatprg=
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=-1
setlocal include=^\\s*\\(from\\|import\\)
setlocal includeexpr=substitute(substitute(substitute(v:fname,b:grandparent_match,b:grandparent_sub,''),b:parent_match,b:parent_sub,''),b:child_match,b:child_sub,'g')
setlocal indentexpr=GetPythonIndent(v:lnum)
setlocal indentkeys=0{,0},:,!^F,o,O,e,<:>,=elif,=except
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=pydoc
set linebreak
setlocal linebreak
setlocal nolisp
setlocal lispwords=
set list
setlocal list
setlocal makeencoding=
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=bin,octal,hex
set number
setlocal number
setlocal numberwidth=4
setlocal omnifunc=python3complete#Complete
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
set relativenumber
setlocal relativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal shiftwidth=4
setlocal noshortname
setlocal signcolumn=auto
setlocal nosmartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=
setlocal suffixesadd=.py
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'python'
setlocal syntax=python
endif
setlocal tabstop=8
setlocal tagcase=
setlocal tags=
setlocal termwinkey=
setlocal termwinscroll=10000
setlocal termwinsize=
setlocal textwidth=80
setlocal thesaurus=
setlocal noundofile
setlocal undolevels=-123456
setlocal varsofttabstop=
setlocal vartabstop=
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
silent! normal! zE
let s:l = 1 - ((0 * winheight(0) + 21) / 43)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1
normal! 0
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToO
set winminheight=1 winminwidth=1
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
