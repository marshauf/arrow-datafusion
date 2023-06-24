// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{AsArray, StringArray, StringBuilder};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::{
    arrow::{array::ArrayRef, datatypes::DataType, record_batch::RecordBatch},
    execution::{
        context::{QueryPlanner, SessionState},
        runtime_env::RuntimeEnv,
        TaskContext,
    },
    logical_expr::Volatility,
    physical_expr::PhysicalSortExpr,
    physical_plan::{
        stream::RecordBatchStreamAdapter, DisplayFormatType, Distribution, ExecutionPlan,
        Partitioning, SendableRecordBatchStream, Statistics,
    },
    physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner},
};

use datafusion::prelude::*;
use datafusion::{error::Result, physical_plan::functions::make_scalar_function};
use datafusion_common::{DFSchemaRef, DataFusionError};
use datafusion_expr::{
    Extension, LogicalPlan, UserDefinedLogicalNode, UserDefinedLogicalNodeCore,
};
use datafusion_optimizer::{optimize_children, OptimizerConfig, OptimizerRule};
use std::{
    any::Any,
    fmt::{self, Debug},
    sync::Arc,
};

use futures::{FutureExt, StreamExt};

// This example demonstrates creating an async user defined function ('url-get').
//
// async user defined functions are not supported directly, but they
// can be implemented using DataFusion's plan rewriting and extension
// mechanisms.
//
// In this example a user defined `QueryPlanner` (GetUrlQueryPlanner) and
// an `OptimizerRule` (GetUrlOptimizerRule) are used to rewrite instances
// of `url-get` into a custom user defined extension nodes.
//
// TODO update:
//
// The `QueryPlanner` registers an `ExtensionPlanner` (GetUrlPlanner).
// The `OptimizerRule` (GetUrlOptimizerRule) replaces a `LogicalPlan::Projection` with a `LogicalPlan::Extension`.
// The extension hosts a user defined node (PowNode).
//
// When creating a physical plan for the extension, the node is casted to a user defined execution plan (GetUrlExec) by the `ExtensionPlanner`.
///
// On plan execution the user defined async function (pow) is called with a `RecordBatch`.

// create local execution context with an in-memory table and
// register an user defined QueryPlanner and a OptimizerRule.
fn create_context() -> Result<SessionContext> {
    // define data.
    let urls = StringArray::from(vec![
        "http://example.com/index.html",
        "http://example.com/data.txt",
    ]);

    let batch = RecordBatch::try_from_iter(vec![("url", Arc::new(urls) as _)])?;

    // declare a state with a query planner and an optimizer rule
    let config = SessionConfig::new();
    let runtime = Arc::new(RuntimeEnv::default());
    let state = SessionState::with_config_rt(config, runtime)
        .with_query_planner(Arc::new(GetUrlQueryPlanner {}))
        .add_optimizer_rule(Arc::new(GetUrlOptimizerRule {}));

    let ctx = SessionContext::with_state(state);
    ctx.register_batch("t", batch)?;
    Ok(ctx)
}

// get_url is an async function and thus could do network I/O or any
// other function. In this example it just returns example data, but
// in a real example it could return anything.
async fn get_url(schema: SchemaRef, input: Result<RecordBatch>) -> Result<RecordBatch> {
    let input = input?;
    let urls: &StringArray = input.column(0).as_string();

    // in this example, pretend to fetch the contents from some remote
    // server.
    let mut contents = StringBuilder::new();
    for url in urls.iter() {
        match url {
            Some(url) => {
                // pretend to fetch the actual data async'hronously
                contents.append_value(format!("Remote data from {url}"));
            }
            None => {
                // ignore NULL input rows
                contents.append_null();
            }
        }
    }

    let batch = RecordBatch::try_new(schema, vec![Arc::new(contents.finish()) as _])?;

    Ok(batch)
}

fn register_udf(ctx: &SessionContext) {
    // First, declare the placeholder UDF
    let placeholder_udf = |_args: &[ArrayRef]| {
        Err(datafusion_common::DataFusionError::NotImplemented(
            "Not supposed to be executed.".to_string(),
        ))
    };

    // the function above expects an `ArrayRef`, but DataFusion may pass a scalar to a UDF.
    // thus, we use `make_scalar_function` to decorare the closure so that it can handle both Arrays and Scalar values.
    let placeholder_udf = make_scalar_function(placeholder_udf);

    let placeholder_udf = create_udf(
        // The name by which it will be called
        "get_url",
        // the input argument type. DataFusion will check this
        vec![DataType::Utf8],
        // The output data type
        Arc::new(DataType::Utf8),
        // If the function can be optimized away. Volatile means
        // DataFusion will not rewrite this during plan time.
        Volatility::Volatile,
        // the actual function implementation
        placeholder_udf,
    );

    // at this point, we register the placeholder so it can be called as a normal function
    ctx.register_udf(placeholder_udf);
}

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = create_context()?;
    register_udf(&ctx);

    // Show the input
    println!("Input:");
    ctx.sql("SELECT * from t").await?.show().await?;

    // Call the UDF from SQL
    println!("Output");
    ctx.sql("SELECT get_url(url) FROM t").await?.show().await?;

    // Show the plan
    println!("Output");
    ctx.sql("EXPLAIN SELECT get_url(url) FROM t")
        .await?
        .show()
        .await?;

    Ok(())
}

struct GetUrlQueryPlanner {}

#[async_trait]
impl QueryPlanner for GetUrlQueryPlanner {
    // Given a `LogicalPlan` created from above, create an `ExecutionPlan` suitable for execution
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Teach the default physical planner how to plan Pow nodes.
        let physical_planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
                GetUrlPlanner {},
            )]);
        // Delegate most work of physical planning to the default physical planner
        physical_planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

// Physical planner for UrlPlanner nodes
struct GetUrlPlanner {}

#[async_trait]
impl ExtensionPlanner for GetUrlPlanner {
    // Create a physical plan for an extension node
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        Ok(
            if let Some(pow_node) = node.as_any().downcast_ref::<GetUrlNode>() {
                Some(Arc::new(GetUrlExec {
                    schema: pow_node.schema.clone(),
                    inputs: physical_inputs.to_vec(),
                }))
            } else {
                None
            },
        )
    }
}

struct GetUrlOptimizerRule {}

impl OptimizerRule for GetUrlOptimizerRule {
    fn try_optimize(
        &self,
        plan: &LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Option<LogicalPlan>> {
        // recurse down and optimize children first, if possible
        let optimized_plan = optimize_children(self, plan, config)?;

        // Get a reference either to the  optimized plan or the input.
        let optimized_plan = optimized_plan.as_ref().unwrap_or(plan);

        // rewrite LogicalPlan::Projection into an extension node.
        match optimized_plan {
            LogicalPlan::Projection(proj) => {
                Ok(Some(LogicalPlan::Extension(Extension {
                    node: Arc::new(GetUrlNode {
                        schema: proj.schema.clone(),
                        input: (*proj.input).clone(),
                        expr: proj.expr.clone(),
                    }),
                })))
            }
            // no rewrite is possible
            _ => Ok(None),
        }
    }

    fn name(&self) -> &str {
        "get_url rewriter"
    }
}

/// Implements a custom DataFusion LogicalPlan node that invokes the
#[derive(PartialEq, Eq, Hash)]
struct GetUrlNode {
    input: LogicalPlan,
    schema: DFSchemaRef,
    expr: Vec<Expr>,
}

impl Debug for GetUrlNode {
    // For PowNode, use explain format for the Debug format. Other types of nodes may
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        UserDefinedLogicalNodeCore::fmt_for_explain(self, f)
    }
}

impl UserDefinedLogicalNodeCore for GetUrlNode {
    fn name(&self) -> &str {
        "get_url"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        self.expr.clone()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "pow expr.len({})", self.expr.len())
    }

    fn from_template(&self, exprs: &[Expr], inputs: &[LogicalPlan]) -> Self {
        Self {
            schema: self.schema.clone(),
            input: inputs[0].clone(),
            expr: exprs.to_vec(),
        }
    }
}

struct GetUrlExec {
    schema: DFSchemaRef,
    inputs: Vec<Arc<dyn ExecutionPlan>>,
}

impl Debug for GetUrlExec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GetUrlExec")
    }
}

#[async_trait]
impl ExecutionPlan for GetUrlExec {
    // Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        (*self.schema).clone().into()
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        None
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        self.inputs.clone()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(GetUrlExec {
            schema: self.schema.clone(),
            inputs: children,
        }))
    }

    // Execute the specified partition and return an stream of RecordBatch
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if 0 != partition {
            return Err(DataFusionError::Internal(format!(
                "GetUrlExec invalid partition {partition}"
            )));
        }

        let schema_captured = self.schema();

        // Inovke the
        let s = self.inputs[0]
            .execute(partition, context)?
            .flat_map(move |b| get_url(schema_captured.clone(), b).into_stream());

        let s = RecordBatchStreamAdapter::new(self.schema(), s);
        Ok(Box::pin(s))
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            // Needed when merged to main
            //DisplayFormatType::Default | DisplayFormatType::Verbose => {
            DisplayFormatType::Default => {
                write!(f, "GetUrlExec")
            }
        }
    }

    fn statistics(&self) -> Statistics {
        // to improve the optimizability of this plan
        // better statistics inference could be provided
        Statistics::default()
    }
}
