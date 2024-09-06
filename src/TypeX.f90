!THIS PACKAGE IS DEDICATED TO CREATE USER DEFINED FILETYPES TO ACCOMODATE CUDA INTERFACE.
!WRITTEN BY KALITH M ISMAIL, SEPTEMBER-2023, CFM, UPV-EHU, SPAIN.

MODULE TYPEX

    ! Define a derived type to represent 1-dimension array (DIM1X)
    TYPE DIM1X
        INTEGER :: X
    END TYPE DIM1X

    ! Define a derived type to represent 2-dimension array (DIM2X)
    TYPE DIM2X
        INTEGER :: X, Y
    END TYPE DIM2X

    ! Define a derived type to represent 3-dimension array (DIM3X)
    TYPE DIM3X
        INTEGER :: X, Y, Z
    END TYPE DIM3X

    ! Define a derived type to represent 4-dimension array (DIM4X)
    TYPE DIM4X
        INTEGER :: W, X, Y, Z
    END TYPE DIM4X

END MODULE TYPEX