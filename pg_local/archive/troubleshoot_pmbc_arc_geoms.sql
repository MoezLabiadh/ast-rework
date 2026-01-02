--POLYGON ((1147509.4917106307111681 472343.1888352958485484, 1160503.1025425037369132 466791.7045031990855932, 1168562.8671169437002391 457458.5914331274107099, 1156277.8705993380863219 460632.2013551639392972, 1154879.6091481007169932 458152.6277052452787757, 1153969.5882088311482221 458343.1237568780779839, 1151071.4373600494582206 461188.1351920440793037, 1147509.4917106307111681 472343.1888352958485484))


WITH validated_parcels AS (
  SELECT *
  FROM WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_FA_SVW b
  WHERE SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT(b.SHAPE, 0.5) = 'TRUE'
)

SELECT b.LEGAL_DESCRIPTION,
       b.PID,
       b.OWNER_TYPE,
       CASE WHEN SDO_GEOM.SDO_DISTANCE(b.SHAPE, 
                    SDO_GEOMETRY(:wkb_aoi, 3005), 0.5) = 0 
            THEN 'INTERSECT' 
            ELSE 'Within ' || TO_CHAR(0) || ' m'
       END AS RESULT,
       SDO_UTIL.TO_WKTGEOMETRY(b.SHAPE) SHAPE

FROM validated_parcels b

WHERE SDO_WITHIN_DISTANCE (b.SHAPE, 
                           SDO_GEOMETRY(:wkb_aoi, 3005),
                           'distance = 0') = 'TRUE';
                           
                           
                           
SELECT b.PID, b.PIN,
       b.OWNER_TYPE,
       SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT(b.SHAPE, 0.5) AS validation_result,
       SDO_UTIL.TO_WKTGEOMETRY(b.SHAPE) SHAPE
FROM WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_FA_SVW b
WHERE SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT(b.SHAPE, 0.5) LIKE '%13347%'
  AND SDO_RELATE(b.SHAPE, 
                 SDO_GEOMETRY(:wkt_geometry, 3005), 
                 'mask=ANYINTERACT') = 'TRUE'
ORDER BY b.PID;








SELECT b.LEGAL_DESCRIPTION,b.PID,b.OWNER_TYPE,
       
       CASE WHEN SDO_GEOM.SDO_DISTANCE(SDO_GEOM.SDO_ARC_DENSIFY(b.SHAPE, 0.005), SDO_GEOMETRY(:wkb_aoi, 3005), 0.5) = 0 
        THEN 'INTERSECT' 
         ELSE 'Within ' || TO_CHAR(0) || ' m'
          END AS RESULT,
          
       SDO_UTIL.TO_WKTGEOMETRY(SDO_GEOM.SDO_ARC_DENSIFY(b.SHAPE, 0.005)) SHAPE

FROM WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_FA_SVW b

WHERE SDO_WITHIN_DISTANCE (SDO_GEOM.SDO_ARC_DENSIFY(b.SHAPE, 0.005), 
                           SDO_GEOMETRY(:wkb_aoi, 3005),'distance = 0') = 'TRUE';
                           
                           
                           
                           
SELECT b.LEGAL_DESCRIPTION,
       b.PID,
       b.OWNER_TYPE,
       CASE WHEN SDO_GEOM.SDO_DISTANCE(
                    SDO_GEOM.SDO_ARC_DENSIFY(
                      b.SHAPE, 
                      m.DIMINFO,
                      'arc_tolerance=0.5'
                    ),
                    SDO_GEOMETRY(:wkb_aoi, 3005), 
                    0.5) = 0 
            THEN 'INTERSECT' 
            ELSE 'Within ' || TO_CHAR(0) || ' m'
       END AS RESULT,
       SDO_UTIL.TO_WKTGEOMETRY(
         SDO_GEOM.SDO_ARC_DENSIFY(
           b.SHAPE,
           m.DIMINFO,
           'arc_tolerance=0.5'
         )
       ) AS SHAPE

FROM WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_FA_SVW b
JOIN USER_SDO_GEOM_METADATA m 
  ON m.TABLE_NAME = 'PMBC_PARCEL_FABRIC_POLY_FA_SVW'
  AND m.COLUMN_NAME = 'SHAPE'

WHERE SDO_WITHIN_DISTANCE (
        SDO_GEOM.SDO_ARC_DENSIFY(
          b.SHAPE,
          m.DIMINFO,
          'arc_tolerance=0.5'
        ),
        SDO_GEOMETRY(:wkb_aoi, 3005),
        'distance = 0') = 'TRUE';                           

SELECT b.PID,
       SDO_GEOM.SDO_ARC_DENSIFY(b.SHAPE, m.DIMINFO, 'arc_tolerance=0.5') AS densified
FROM WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_FA_SVW b
JOIN ALL_SDO_GEOM_METADATA m  -- Try ALL_ instead of USER_
  ON m.TABLE_NAME = 'PMBC_PARCEL_FABRIC_POLY_FA_SVW'
  AND m.COLUMN_NAME = 'SHAPE'
WHERE ROWNUM <= 10;



SELECT b.LEGAL_DESCRIPTION,
       b.PID,
       b.OWNER_TYPE,
       CASE WHEN SDO_GEOM.SDO_DISTANCE(
                    SDO_GEOM.SDO_BUFFER(
                      SDO_GEOM.SDO_SELF_UNION(b.SHAPE, 0.5), 
                      0, 
                      0.5
                    ), 
                    SDO_GEOMETRY(:wkb_aoi, 3005), 
                    0.5) = 0 
            THEN 'INTERSECT' 
            ELSE 'Within ' || TO_CHAR(0) || ' m'
       END AS RESULT,
       SDO_UTIL.TO_WKTGEOMETRY(
         SDO_GEOM.SDO_BUFFER(
           SDO_GEOM.SDO_SELF_UNION(b.SHAPE, 0.5), 
           0, 
           0.5
         )
       ) AS SHAPE

FROM WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_FA_SVW b

WHERE SDO_WITHIN_DISTANCE (
        SDO_GEOM.SDO_BUFFER(
          SDO_GEOM.SDO_SELF_UNION(b.SHAPE, 0.5), 
          0, 
          0.5
        ),
        SDO_GEOMETRY(:wkb_aoi, 3005),
        'distance = 0') = 'TRUE';
        
        
SELECT s.SHAPE.sdo_srid SP_REF
--FROM WHSE_FOREST_TENURE.FTEN_MAP_NOTATN_POINTS_SVW s
FROM WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_FA_SVW s
WHERE rownum = 1
;        
        
 -- THIS works but eliminates non-valid geometries
 SELECT b.LEGAL_DESCRIPTION,
       b.PID,
       b.OWNER_TYPE,
       CASE WHEN SDO_GEOM.SDO_DISTANCE(b.SHAPE, 
                                        SDO_GEOMETRY(:wkb_aoi, 3005), 
                                        0.5) = 0 
            THEN 'INTERSECT' 
            ELSE 'Within ' || TO_CHAR(0) || ' m'
       END AS RESULT,
       SDO_UTIL.TO_WKTGEOMETRY(b.SHAPE) AS SHAPE

FROM WHSE_CADASTRE.PMBC_PARCEL_FABRIC_POLY_FA_SVW b

WHERE SDO_GEOM.VALIDATE_GEOMETRY_WITH_CONTEXT(b.SHAPE, 0.5) = 'TRUE'
  AND SDO_WITHIN_DISTANCE (b.SHAPE,
                           SDO_GEOMETRY(:wkb_aoi, 3005),
                           'distance = 0') = 'TRUE'
ORDER BY b.PID;

SDO_UTIL.RECTIFY_GEOMETRY(b.{geom_col}, 0.05)

SELECT * FROM  WHSE_FOREST_TENURE.FTEN_MAP_NOTATN_POINTS_SVW;