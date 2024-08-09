@schema
class User(dj.Lookup):
    definition = """
    # Users of the light sheet microscope
    username : varchar(20)      # user in the lab
    ---
    princeton_email       : varchar(50)
    """

@schema
class Request(dj.Manual):
    definition = """ # The highest level table for handling user requests to the Core Facility 
    -> User  
    request_name                    :   varchar(64)
    ----
    -> User.proj(requested_by='username') # defines a new column here called "requested_by" whose value must be one of the "username" entries in the User() table    
    -> [nullable] User.proj(auditor='username')  # defines a new column here called "auditor" whose value must be one of the "username" entries in the User() table    
    date_submitted                  :   date     # The date it was submitted as a request
    time_submitted                  :   time     # The time it was submitted as a request
    labname                         :   varchar(50)
    correspondence_email = ''       :   varchar(100)
    description                     :   varchar(250)
    species                         :   varchar(50)
    number_of_samples               :   tinyint
    testing = 0                     :   boolean
    is_archival = 0                 :   boolean
    raw_data_retention_preference = NULL  :   enum("important","kind of important","not important","not sure")
    sent_processing_email = 0       :   boolean
    """  
    class Sample(dj.Part):
        definition = """ # Samples from a request
        -> Request
        sample_name                  :   varchar(64)                
        ----
        subject_fullname = NULL      :   varchar(64)
        """ 

    class ClearingBatch(dj.Part):
        definition = """ # Samples from a particular request
        -> Request
        clearing_batch_number        :   tinyint
        ----
        clearing_protocol            :   enum("iDISCO+_immuno","iDISCO abbreviated clearing","iDISCO abbreviated clearing (rat)","uDISCO","uDISCO (rat)","iDISCO_EdU","experimental")
        antibody1 = ''               :   varchar(100)
        antibody2 = ''               :   varchar(100)
        antibody1_lot = NULL         :   varchar(64)
        antibody2_lot = NULL         :   varchar(64)
        clearing_progress            :   enum("incomplete","in progress","complete")
        number_in_batch              :   tinyint
        perfusion_date = NULL        :   date
        expected_handoff_date = NULL :   date
        -> [nullable] User.proj(clearer='username') # defines a new column here called "clearer" whose value must be either None or one of the "username" entries in the User() table
        notes_for_clearer = ""       :   varchar(8192)                
        link_to_clearing_spreadsheet = NULL : varchar(256)
        """  
    
    class ClearingBatchSample(dj.Part):
        definition = """ # Samples in a ClearingBatch
        -> master.Sample                
        -> master.ClearingBatch
        ----
        clearing_protocol            :   enum("iDISCO+_immuno","iDISCO abbreviated clearing","iDISCO abbreviated clearing (rat)","uDISCO","uDISCO (rat)","iDISCO_EdU","experimental")
        antibody1 = ''               :   varchar(100)
        antibody2 = ''               :   varchar(100)
        """

    class ImagingBatch(dj.Part):
        definition = """ # Batch of Samples to image the same way
        -> master.ClearingBatch
        imaging_batch_number                      :   tinyint
        imaging_request_number                    :   tinyint
        ----
        -> [nullable] User.proj(imager='username') # defines a new column here called "imager" whose value must be one of the "username" entries in the User() table
        number_in_imaging_batch                   :   tinyint # date that the imaging form was submitted by the imager
        imaging_request_date_submitted            :   date # date that the user submitted the request for imaging
        imaging_request_time_submitted            :   time # time that the user submitted the request for imaging
        imaging_performed_date = NULL             :   date # date that the imaging form was submitted by the imager
        imaging_progress                          :   enum("incomplete","in progress","complete")
        imaging_dict                              :   blob
        """

    class ImagingBatchSample(dj.Part):
        definition = """ # Samples in an ImagingBatch
        -> master.Sample
        -> master.ImagingBatch
        ----
        """
    
    class ImagingRequest(dj.Part):
        definition = """ # Imaging request
        -> master.Sample
        imaging_request_number                    :   tinyint
        ----
        -> [nullable] User.proj(imager='username') # defines a new column here called "imager" whose value must be one of the "username" entries in the User() table
        imaging_request_date_submitted            :   date # date that the user submitted the request for imaging
        imaging_request_time_submitted            :   time # time that the user submitted the request for imaging
        imaging_performed_date = NULL             :   date # date that the imaging form was submitted by the imager
        imaging_progress                          :   enum("incomplete","in progress","complete")
        imaging_skipped = NULL                    :   boolean # 1 if this sample skipped, 0 or NULL if not skipped
        """

    class ImagingResolutionRequest(dj.Part):
        definition = """ # Imaging parameters for a channel, belonging to a sample
        -> master.ImagingRequest
        image_resolution                          :   enum("1.3x","4x","1.1x","2x","3.6x","15x")
        ----        
        microscope = NULL                         :   enum("LaVision","SmartSPIM")
        notes_for_imager = ""                     :   varchar(1024)
        notes_from_imaging = ""                   :   varchar(1024)
        """
    
    class ImagingChannel(dj.Part):
        definition = """ # Imaging parameters for a channel, belonging to a sample
        -> master.ImagingResolutionRequest
        channel_name                                            :   varchar(64)                
        ventral_up = 0                                          :   boolean # whether brain was flipped upside down to be imaged
        ----
        imaging_date = NULL                                     :   date 
        zoom_body_magnification = NULL                          :   float # only applicable for 2x
        left_lightsheet_used = 1                                :   boolean
        right_lightsheet_used = 1                               :   boolean
        registration = 0                                        :   boolean
        injection_detection = 0                                 :   boolean
        probe_detection = 0                                     :   boolean
        cell_detection = 0                                      :   boolean
        generic_imaging = 0                                     :   boolean
        pixel_type = NULL                                       :   varchar(32)
        image_orientation                                       :   enum("sagittal","coronal","horizontal") # how the imager imaged the sample. Most of the time will be horizontal
        numerical_aperture = NULL                               :   float # it is not always recorded in metadata so those times it will be NULL
        tiling_scheme = '1x1'                                   :   char(3)
        tiling_overlap = 0.0                                    :   float
        z_step = 10                                             :   float # distance between z planes in microns
        number_of_z_planes = NULL                               :   smallint unsigned
        rawdata_subfolder = NULL                                :   varchar(512)
        imspector_channel_index = NULL                          :   tinyint    # refers to multi-channel imaging - 0 if first (or only) channel in rawdata_subfolder, 1 if second, 2 if third, ...
        left_lightsheet_precomputed_spock_jobid = NULL          :   varchar(32)
        left_lightsheet_precomputed_spock_job_progress = NULL   :   enum("NOT_SUBMITTED","SUBMITTED","COMPLETED","FAILED","RUNNING","PENDING","BOOT_FAIL","CANCELLED","DEADLINE","OUT_OF_MEMORY","REQUEUED"," RESIZING","REVOKED","SUSPENDED","TIMEOUT")
        right_lightsheet_precomputed_spock_jobid = NULL         :   varchar(32)
        right_lightsheet_precomputed_spock_job_progress = NULL  :   enum("NOT_SUBMITTED","SUBMITTED","COMPLETED","FAILED","RUNNING","PENDING","BOOT_FAIL","CANCELLED","DEADLINE","OUT_OF_MEMORY","REQUEUED"," RESIZING","REVOKED","SUSPENDED","TIMEOUT")
        """

   
    class ProcessingRequest(dj.Part):
        definition = """ # Processing request - this needs to exist because for each imaging request there can be multiple processing requests 
        -> master.ImagingRequest
        processing_request_number                    :   tinyint
        ----
        -> [nullable] User.proj(processor='username') # defines column "processor" whose value must be one of the "username" entries in the User() table
        processing_request_date_submitted            :   date # date that the user submitted the request for processing
        processing_request_time_submitted            :   time # time that the user submitted the request for processing
        processing_performed_date = NULL             :   date # date that the processing form was submitted by the processor
        processing_progress                          :   enum("incomplete","running","failed", "complete")
        """

    class ProcessingResolutionRequest(dj.Part):
        definition = """ # Processing parameters at the image resolution level for a given ProcessingRequest(). These represent spock jobs
        -> master.ProcessingRequest
        image_resolution                          :   enum("1.3x","4x","1.1x","2x","3.6x","15x")
        ventral_up = 0                            :   boolean # whether brain was flipped upside down to be imaged
        ----        
        atlas_name                                :   enum("allen_2017","allen_2011","princeton_mouse_atlas","paxinos")
        final_orientation                         :   enum("sagittal","coronal","horizontal")
        notes_for_processor = ""                  :   varchar(1024)
        notes_from_processing = ""                :   varchar(1024) 
        lightsheet_pipeline_spock_jobid = NULL            :   varchar(16)  # the jobid from the final step in the light sheet processing pipeline for LaVision
        lightsheet_pipeline_spock_job_progress = NULL     :   enum("NOT_SUBMITTED","SUBMITTED","COMPLETED","FAILED","RUNNING","PENDING","BOOT_FAIL","CANCELLED","DEADLINE","OUT_OF_MEMORY","REQUEUED"," RESIZING","REVOKED","SUSPENDED","TIMEOUT") # the spock job status code for the final step in the light sheet processing pipeline
        brainpipe_commit = NULL                   : char(7) # the commit that is checked out on the machine at the time the job was submitted 
        """

    class ProcessingChannel(dj.Part):
        definition = """ # Processing parameters for a channel. There can be more than one purpose for a single channel, hence why lightsheet_channel_str is a primary key
        -> master.ImagingChannel
        -> master.ProcessingResolutionRequest
        lightsheet_channel_str                                        :   enum("regch","injch","cellch","gench")
        ----
        imspector_version = ''                                        :   varchar(128)
        datetime_processing_started                                   :   datetime
        datetime_processing_completed = NULL                          :   datetime
        intensity_correction = 1                                      :   boolean
        metadata_xml_string = NULL                                    :   mediumblob # The entire metadata xml string. Sometimes it is not available so those times it will be NULL
        left_lightsheet_stitched_precomputed_spock_jobid = NULL          :   varchar(32)
        left_lightsheet_stitched_precomputed_spock_job_progress = NULL   :   enum("NOT_SUBMITTED","SUBMITTED","COMPLETED","FAILED","RUNNING","PENDING","BOOT_FAIL","CANCELLED","DEADLINE","OUT_OF_MEMORY","REQUEUED"," RESIZING","REVOKED","SUSPENDED","TIMEOUT")
        right_lightsheet_stitched_precomputed_spock_jobid = NULL         :   varchar(32)
        right_lightsheet_stitched_precomputed_spock_job_progress = NULL  :   enum("NOT_SUBMITTED","SUBMITTED","COMPLETED","FAILED","RUNNING","PENDING","BOOT_FAIL","CANCELLED","DEADLINE","OUT_OF_MEMORY","REQUEUED"," RESIZING","REVOKED","SUSPENDED","TIMEOUT")
        blended_precomputed_spock_jobid = NULL                        : varchar(32)
        blended_precomputed_spock_job_progress = NULL                 : enum("NOT_SUBMITTED","SUBMITTED","COMPLETED","FAILED","RUNNING","PENDING","BOOT_FAIL","CANCELLED","DEADLINE","OUT_OF_MEMORY","REQUEUED"," RESIZING","REVOKED","SUSPENDED","TIMEOUT")
        downsized_precomputed_spock_jobid = NULL                      : varchar(32)
        downsized_precomputed_spock_job_progress = NULL               : enum("NOT_SUBMITTED","SUBMITTED","COMPLETED","FAILED","RUNNING","PENDING","BOOT_FAIL","CANCELLED","DEADLINE","OUT_OF_MEMORY","REQUEUED"," RESIZING","REVOKED","SUSPENDED","TIMEOUT")
        registered_precomputed_spock_jobid = NULL                     : varchar(32)
        registered_precomputed_spock_job_progress = NULL              : enum("NOT_SUBMITTED","SUBMITTED","COMPLETED","FAILED","RUNNING","PENDING","BOOT_FAIL","CANCELLED","DEADLINE","OUT_OF_MEMORY","REQUEUED"," RESIZING","REVOKED","SUSPENDED","TIMEOUT")
        """
    
    class SmartspimStitchedChannel(dj.Part):
        definition = """ # Record of which SmartSPIM channels have undergone stitching
        -> master.ImagingChannel
        ----
        datetime_stitching_started                                    :   datetime
        datetime_stitching_completed = NULL                           :   datetime
        smartspim_stitching_spock_jobid = NULL                        :   varchar(16)  # the jobid from the final stitching step for SmartSPIM
        smartspim_stitching_spock_job_progress = NULL                 :   enum("NOT_SUBMITTED","SUBMITTED","COMPLETED","FAILED","RUNNING","PENDING","BOOT_FAIL","CANCELLED","DEADLINE","OUT_OF_MEMORY","REQUEUED"," RESIZING","REVOKED","SUSPENDED","TIMEOUT") # the spock job status code for this smartspim stitching task
        brainpipe_commit = NULL                                       :   char(7) # the commit that is checked out on the machine at the time the job was submitted 
        """  

    class IdiscoAbbreviatedClearing(dj.Part): 
        definition = """ # iDISCO abbreviated clearing table
        -> master.ClearingBatch           
        ----
        clearing_protocol            :   enum("iDISCO+_immuno","iDISCO abbreviated clearing","iDISCO abbreviated clearing (rat)","uDISCO","uDISCO (rat)","iDISCO_EdU","experimental")
        antibody1 = ''               :   varchar(100)
        antibody2 = ''               :   varchar(100)
        exp_notes = ""                                           :   varchar(500)  # Note anything unusual that happened during request handling that could affect clearing
        time_pbs_wash1 = NULL                                    :   datetime
        pbs_wash1_notes = ""                                     :   varchar(250)
        time_pbs_wash2 = NULL                                    :   datetime
        pbs_wash2_notes = ""                                     :   varchar(250)
        time_pbs_wash3 = NULL                                    :   datetime
        pbs_wash3_notes = ""                                     :   varchar(250)
        time_dehydr_methanol_20percent_wash1 = NULL                 :   datetime
        dehydr_methanol_20percent_wash1_notes = ""                  :   varchar(250)
        time_dehydr_methanol_40percent_wash1 = NULL                 :   datetime
        dehydr_methanol_40percent_wash1_notes = ""                  :   varchar(250)
        time_dehydr_methanol_60percent_wash1 = NULL                 :   datetime
        dehydr_methanol_60percent_wash1_notes = ""                  :   varchar(250)
        time_dehydr_methanol_80percent_wash1 = NULL                 :   datetime
        dehydr_methanol_80percent_wash1_notes = ""                  :   varchar(250)
        time_dehydr_methanol_100percent_wash1 = NULL                :   datetime
        dehydr_methanol_100percent_wash1_notes = ""                 :   varchar(250)
        time_dehydr_methanol_100percent_wash2 = NULL                :   datetime
        dehydr_methanol_100percent_wash2_notes = ""                 :   varchar(250)
        time_dehydr_dcm_66percent_methanol_33percent = NULL         :   datetime
        dehydr_dcm_66percent_methanol_33percent_notes = ""          :   varchar(250)
        time_dehydr_dcm_wash1 = NULL                             :   datetime
        dehydr_dcm_wash1_notes = ""                              :   varchar(250)
        time_dehydr_dcm_wash2 = NULL                             :   datetime
        dehydr_dcm_wash2_notes = ""                              :   varchar(250)
        time_dehydr_dbe_wash1 = NULL                             :   datetime
        dehydr_dbe_wash1_notes = ""                              :   varchar(250)
        time_dehydr_dbe_wash2 = NULL                             :   datetime
        dehydr_dbe_wash2_notes = ""                              :   varchar(250)
        clearing_notes = ""                                      :   varchar(500)
        """