from library.database_model.slide import Section

class SectionsController():
    """Class for controlling sections
    """

    def get_sections(self, animal: str, channel: int, debug: bool = False) -> list:
        """The sections table is a view and it is already filtered by active and file_status = 'good'
        The ordering is important. This needs to come from the histology table.

        :param animal: the animal to query
        :param channel: 1 or 2 or 3.
        :param debug: whether to print the raw SQL query

        :returns: list of sections in order
        """
        slide_orderby = self.histology.side_sectioned_first
        scene_order_by = self.histology.scene_order

        if slide_orderby == 'Right' and scene_order_by == 'DESC':
            query = self.session.query(Section).filter(Section.prep_id == animal)\
                .filter(Section.channel == channel)\
                .order_by(Section.slide_physical_id.desc())\
                .order_by(Section.scene_number.desc())
        elif slide_orderby == 'Left' and scene_order_by == 'ASC':
            query = self.session.query(Section).filter(Section.prep_id == animal)\
                .filter(Section.channel == channel)\
                .order_by(Section.slide_physical_id.asc())\
                .order_by(Section.scene_number.asc())
        elif slide_orderby == 'Left' and scene_order_by == 'DESC':
            query = self.session.query(Section).filter(Section.prep_id == animal)\
                .filter(Section.channel == channel)\
                .order_by(Section.slide_physical_id.asc())\
                .order_by(Section.scene_number.desc())
        elif slide_orderby == 'Right' and scene_order_by == 'ASC':
            query = self.session.query(Section).filter(Section.prep_id == animal)\
                .filter(Section.channel == channel)\
                .order_by(Section.slide_physical_id.desc())\
                .order_by(Section.scene_number.asc())
        else:
            print('Using default order by')
            query = self.session.query(Section).filter(Section.prep_id == animal)\
                .filter(Section.channel == channel)\
                .order_by(Section.slide_physical_id.asc())\
                .order_by(Section.scene_number.asc())

        if debug: # Print the raw SQL query
            print(f'RAW SQL: {str(query.statement.compile(compile_kwargs={"literal_binds": True}))}')

        sections = query.all()
        return sections


    def get_section_count(self, animal):
        count = self.session.query(Section)\
            .filter(Section.prep_id == animal)\
            .filter(Section.channel == 1)\
            .count() 
        return count