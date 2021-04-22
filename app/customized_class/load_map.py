def load_source_map(map_):
    
    '''
    This function create a map source object which is possible embedding in an iframe html object.
    
    '''
    source = map_.get_root().render().replace('"', '&quot;')
    
    return source