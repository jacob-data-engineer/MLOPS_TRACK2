#!/usr/bin/env bash
# One-time setup for the Track2 SQL tab (2026-10-28). Run from Git Bash in
# the T2split folder:   bash scripts/setup_sql_tab.sh
#
# What it does, in order -- every step was approved in the chat:
#   1. A random password per database for `track2_readonly`, kept as an
#      encrypted SSM parameter (/track2/pg-query/<db>/readonly-password).
#   2. The two query Lambdas, one inside each database's VPC, attached to
#      the security group that database already lets in (nothing of theirs
#      is changed). Role track2-pg-query-lambda already exists.
#   3. Setup inside each database: reads the ADMIN password from the RDS
#      secret, passes it in ONE invocation (never stored), and the function
#      creates track2_readonly with SELECT only. Then ALLOW_BOOTSTRAP is
#      removed so it cannot be done again.
#   4. Lets the app (role track2-yolo-review-host) invoke those two
#      functions and read the databases' status.
#   5. Stops the preview database again (it was started for step 3).
#
# Safe to re-run: existing parameters, functions and the role are reused;
# the read-only user's password is re-set to the stored one.
set -euo pipefail
export MSYS_NO_PATHCONV=1          # keep /track2/... from becoming C:/...
REGION=ap-southeast-1
ACCOUNT=125192671932
ROLE_ARN=arn:aws:iam::$ACCOUNT:role/track2-pg-query-lambda
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# A bare `python` in Git Bash is the Windows Store alias and fails.
PY="${PY:-/c/Projects/MLOPS_TRACK2/.venv/Scripts/python.exe}"
ZIP="$HERE/terraform/app/.build/pg_query.zip"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# The AWS CLI here is a Windows program: it needs Windows paths for the
# files it reads or writes (path conversion is off, see the top).
win() { cygpath -w "$1" 2>/dev/null || echo "$1"; }

# key  function  host  database  subnets  security-group  admin-secret  instance
DBS=(
  "violation-review|track2-pg-query-violation-review|violation-review-postgres.cf08ywocgcwq.ap-southeast-1.rds.amazonaws.com|violation_review|subnet-07ae59d94573b9708,subnet-0010435bfb603427a,subnet-06e4c92be8027e860|sg-0ef3ab2a18f121f50|rds!db-2b0880e2-7730-47f1-878b-5391095f3f5b|violation-review-postgres"
  "jpui-preview|track2-pg-query-jpui-preview|jpui-preview-crop-authority.cf08ywocgcwq.ap-southeast-1.rds.amazonaws.com|violation_review_preview|subnet-0e39f44b2e9fd0eaf,subnet-04752d695dc4406c4|sg-0ef2fc81fd0ae946b|rds!db-37099466-58b2-4fbc-9776-27011d24ef0a|jpui-preview-crop-authority"
)

if [ ! -f "$ZIP" ]; then
  echo "Packaging the function ..."
  rm -rf "$HERE/terraform/app/.build/pg_query"
  uv pip install --target "$HERE/terraform/app/.build/pg_query" --python-platform x86_64-manylinux2014 --python-version 3.12 pg8000 >/dev/null
  cp "$HERE/terraform/app/lambda_src/pg_query.py" "$HERE/terraform/app/.build/pg_query/"
  (cd "$HERE/terraform/app/.build/pg_query" && "$PY" -c "import shutil; shutil.make_archive('../pg_query','zip','.')")
fi
ZIPW="$(cygpath -w "$ZIP" 2>/dev/null || echo "$ZIP")"

env_file() {  # host database password bootstrap(0|1) -> path of a JSON env file
  "$PY" - "$1" "$2" "$3" "$4" > "$TMP/env.json" <<'PY'
import json, sys
host, db, pw, boot = sys.argv[1:]
env = {"PGHOST": host, "PGPORT": "5432", "PGDATABASE": db,
       "PGUSER": "track2_readonly", "PGPASSWORD": pw}
if boot == "1":
    env["ALLOW_BOOTSTRAP"] = "1"
print(json.dumps({"Variables": env}))
PY
  echo "$TMP/env.json"
}

for row in "${DBS[@]}"; do
  IFS='|' read -r KEY FN HOST DB SUBNETS SG SECRET INSTANCE <<< "$row"
  echo; echo "=== $KEY"

  # 1. the read-only password
  PARAM="/track2/pg-query/$KEY/readonly-password"
  if ! PW=$(aws ssm get-parameter --region $REGION --name "$PARAM" --with-decryption --query Parameter.Value --output text 2>/dev/null); then
    PW=$("$PY" -c "import secrets,string; a=string.ascii_letters+string.digits; print(''.join(secrets.choice(a) for _ in range(40)))")
    aws ssm put-parameter --region $REGION --name "$PARAM" --type SecureString --value "$PW" \
      --description "track2_readonly login for the Track2 SQL tab ($KEY)" >/dev/null
    echo "stored a new read-only password in $PARAM"
  fi

  # 2. the function, with setup switched on
  ENV=$(env_file "$HOST" "$DB" "$PW" 1)
  if aws lambda get-function --region $REGION --function-name "$FN" >/dev/null 2>&1; then
    aws lambda update-function-code --region $REGION --function-name "$FN" --zip-file "fileb://$ZIPW" >/dev/null
    aws lambda wait function-updated --region $REGION --function-name "$FN"
    aws lambda update-function-configuration --region $REGION --function-name "$FN" --environment "file://$(win "$ENV")" >/dev/null
  else
    aws lambda create-function --region $REGION --function-name "$FN" --runtime python3.12 \
      --handler pg_query.lambda_handler --role "$ROLE_ARN" --zip-file "fileb://$ZIPW" \
      --timeout 60 --memory-size 256 \
      --vpc-config "SubnetIds=$SUBNETS,SecurityGroupIds=$SG" \
      --environment "file://$(win "$ENV")" --tags Project=track2 \
      --description "Track2 SQL tab: read-only queries against $DB" >/dev/null
  fi
  aws lambda wait function-active-v2 --region $REGION --function-name "$FN" 2>/dev/null \
    || aws lambda wait function-active --region $REGION --function-name "$FN"
  aws lambda wait function-updated --region $REGION --function-name "$FN"
  echo "function $FN ready"

  # 3. create track2_readonly, as admin, once
  STATUS=$(aws rds describe-db-instances --region $REGION --db-instance-identifier "$INSTANCE" --query "DBInstances[0].DBInstanceStatus" --output text)
  if [ "$STATUS" = "stopped" ]; then
    echo "starting $INSTANCE for the setup (stopped again at the end) ..."
    aws rds start-db-instance --region $REGION --db-instance-identifier "$INSTANCE" >/dev/null
    STATUS=starting
  fi
  if [ "$STATUS" != "available" ]; then
    echo "waiting for $INSTANCE ($STATUS) ..."
    aws rds wait db-instance-available --region $REGION --db-instance-identifier "$INSTANCE"
  fi
  aws secretsmanager get-secret-value --region $REGION --secret-id "$SECRET" --query SecretString --output text \
    | "$PY" -c "import json,sys; s=json.load(sys.stdin); print(json.dumps({'action':'bootstrap','admin_user':s['username'],'admin_password':s['password']}))" \
    > "$TMP/boot.json"
  aws lambda invoke --region $REGION --function-name "$FN" --cli-binary-format raw-in-base64-out \
    --payload "file://$(win "$TMP/boot.json")" "$(win "$TMP/out.json")" >/dev/null
  rm -f "$TMP/boot.json"
  echo "setup said: $(cat "$TMP/out.json")"
  grep -q '"ok": true' "$TMP/out.json" || { echo "setup FAILED for $KEY -- stopping." >&2; exit 1; }

  # ...and switch setup off again
  ENV=$(env_file "$HOST" "$DB" "$PW" 0)
  aws lambda update-function-configuration --region $REGION --function-name "$FN" --environment "file://$(win "$ENV")" >/dev/null
  aws lambda wait function-updated --region $REGION --function-name "$FN"

  # a first read-only query, as proof
  echo '{"action":"query","sql":"select current_user, current_setting('"'"'transaction_read_only'"'"')"}' > "$TMP/q.json"
  aws lambda invoke --region $REGION --function-name "$FN" --cli-binary-format raw-in-base64-out \
    --payload "file://$(win "$TMP/q.json")" "$(win "$TMP/out.json")" >/dev/null
  echo "check: $(cat "$TMP/out.json")"
done

# 4. the app may call the two functions and read the databases' status
cat > "$TMP/app-policy.json" <<EOF
{"Version":"2012-10-17","Statement":[
 {"Sid":"InvokeSqlQueryFunctions","Effect":"Allow","Action":"lambda:InvokeFunction",
  "Resource":["arn:aws:lambda:$REGION:$ACCOUNT:function:track2-pg-query-violation-review",
              "arn:aws:lambda:$REGION:$ACCOUNT:function:track2-pg-query-jpui-preview"]},
 {"Sid":"ReadSqlDatabaseStatus","Effect":"Allow","Action":"rds:DescribeDBInstances","Resource":"*"}]}
EOF
aws iam put-role-policy --role-name track2-yolo-review-host --policy-name track2-sql-tab \
  --policy-document "file://$(win "$TMP/app-policy.json")"
echo; echo "app role may now invoke both query functions"

# 5. the preview database was started only for this
aws rds stop-db-instance --region $REGION --db-instance-identifier jpui-preview-crop-authority \
  --query DBInstance.DBInstanceStatus --output text || true
echo "done."
