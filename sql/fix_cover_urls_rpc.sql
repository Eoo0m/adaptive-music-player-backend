CREATE OR REPLACE FUNCTION public.fix_track_cover_image_urls(
    old_host text DEFAULT 'xopigdxsobjbagmazgba.supabase.co',
    new_host text DEFAULT 'yxihvsnrszafmluyjqke.supabase.co',
    dry_run boolean DEFAULT true
)
RETURNS TABLE (
    matched_count bigint,
    updated_count bigint
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    SELECT count(*)
    INTO matched_count
    FROM public.track_embeddings
    WHERE cover_image_url LIKE '%' || old_host || '%';

    IF dry_run THEN
        updated_count := 0;
        RETURN NEXT;
        RETURN;
    END IF;

    UPDATE public.track_embeddings
    SET
        cover_image_url = replace(cover_image_url, old_host, new_host),
        updated_at = now()
    WHERE cover_image_url LIKE '%' || old_host || '%';

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN NEXT;
END;
$$;

REVOKE ALL ON FUNCTION public.fix_track_cover_image_urls(text, text, boolean) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.fix_track_cover_image_urls(text, text, boolean) TO service_role;
