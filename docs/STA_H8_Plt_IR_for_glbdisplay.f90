!=======================================================================
      Program Plt_IR_for_glbdisplay
!=======================================================================
      implicit none
      integer, parameter       :: ix = 2750, iy = 2750
      
      character*120, parameter :: inflLL= 'Proj_Scale050_2750x2750'

      integer      :: i,j,ii,jj
      real         :: IR(ix,iy)
      character*120:: fn

      !! 讀取H8/9衛星觀測資料
      fn = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
      read (  1,'(a120)')fn
      open ( 99, file=trim(fn), access='direct'  &
               , form='unformatted', status='unknown', recl=4*ix*iy) 
      read ( 99,rec=1) ((IR(i,j),i=1,ix),j=iy,1,-1)
      close( 99)

      !! 讀取H8/9 CWA東亞地區pixel經緯度
      open (99, file=trim(inflLL), access='direct'  &
              , form='unformatted', status='unknown', recl=2*4*ix*iy) 
      read (99,rec=1) ((latlon(i,j,1),i=1,ix),j=iy,1,-1),((latlon(i,j,2),i=1,ix),j=iy,1,-1)
      close(99)
